'''override pymilvus' Collection class to fix its bugs & add async.'''

# Copyright (C) 2019-2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

import copy
import json
import pandas as pd
from typing import Dict, List, Optional, Union, overload

from pymilvus.client.abstract import SearchResult
from pymilvus.client.constants import DEFAULT_CONSISTENCY_LEVEL
from pymilvus.client.types import (
    CompactionPlans,
    CompactionState,
    Replica,
    cmp_consistency_level,
    get_consistency_level,
)
from pymilvus.exceptions import (
    AutoIDException,
    DataTypeNotMatchException,
    ExceptionsMessage,
    IndexNotExistException,
    PartitionAlreadyExistException,
    SchemaNotReadyException,
)
from pymilvus.grpc_gen import schema_pb2
from pymilvus.settings import Config
from pymilvus.orm.connections import connections
from pymilvus.orm.constants import UNLIMITED
from pymilvus.orm.index import Index
from pymilvus.orm.partition import Partition
from pymilvus.orm.prepare import Prepare
from pymilvus.orm.schema import (
    CollectionSchema,
    FieldSchema,
    check_insert_schema,
    check_is_row_based,
    check_schema,
    check_upsert_schema,
    construct_fields_from_dataframe,
)
from pymilvus.orm.types import DataType
from pymilvus.orm.utility import _get_connection
from pymilvus.client.asynch import MutationResult, MutationFuture
from pymilvus import Collection as _Collection

from ._iterator import QueryIterator, SearchIterator
from ._handler import AsyncGrpcHandler

class Collection(_Collection):  # inherit to make isinstance(...,_Collection) return True
    
    def __init__(
        self,
        name: str,
        schema: Optional[CollectionSchema] = None,
        using: str = "default",
        **kwargs,
    ) -> None:
        """Constructs a collection by name, schema and other parameters.

        Args:
            name (``str``): the name of collection
            schema (``CollectionSchema``, optional): the schema of collection, defaults to None.
            using (``str``, optional): Milvus connection alias name, defaults to 'default'.
            **kwargs (``dict``):

                * *num_shards (``int``, optional): how many shards will the insert data be divided.
                * *shards_num (``int``, optional, deprecated):
                    how many shards will the insert data be divided.
                * *consistency_level* (``int/ str``)
                    Which consistency level to use when searching in the collection.
                    Options of consistency level: Strong, Bounded, Eventually, Session, Customized.

                    Note: can be overwritten by the same parameter specified in search.

                * *properties* (``dict``, optional)
                    Collection properties.

                * *timeout* (``float``)
                    An optional duration of time in seconds to allow for the RPCs.
                    If timeout is not set, the client keeps waiting until the server
                    responds or an error occurs.


        Raises:
            SchemaNotReadyException: if the schema is wrong.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> fields = [
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=128)
            ... ]
            >>> schema = CollectionSchema(fields=fields)
            >>> prop = {"collection.ttl.seconds": 1800}
            >>> collection = Collection(name="test_collection_init", schema=schema, properties=prop)
            >>> collection.name
            'test_collection_init'
        """
        self._name = name
        self._using = using
        self._kwargs = kwargs
        self._num_shards = None
        conn = self._get_connection()

        has = conn.has_collection(self._name, **kwargs)
        if has:
            resp = conn.describe_collection(self._name, **kwargs)
            s_consistency_level = resp.get("consistency_level", DEFAULT_CONSISTENCY_LEVEL)
            arg_consistency_level = kwargs.get("consistency_level", s_consistency_level)
            if not cmp_consistency_level(s_consistency_level, arg_consistency_level):
                raise SchemaNotReadyException(
                    message=ExceptionsMessage.ConsistencyLevelInconsistent
                )
            server_schema = CollectionSchema.construct_from_dict(resp)
            self._consistency_level = s_consistency_level
            if schema is None:
                self._schema = server_schema
            else:
                if not isinstance(schema, CollectionSchema):
                    raise SchemaNotReadyException(message=ExceptionsMessage.SchemaType)
                if server_schema != schema:
                    raise SchemaNotReadyException(message=ExceptionsMessage.SchemaInconsistent)
                self._schema = schema

        else:
            if schema is None:
                raise SchemaNotReadyException(
                    message=ExceptionsMessage.CollectionNotExistNoSchema % name
                )
            if isinstance(schema, CollectionSchema):
                schema.verify()
                check_schema(schema)
                consistency_level = get_consistency_level(
                    kwargs.get("consistency_level", DEFAULT_CONSISTENCY_LEVEL)
                )

                conn.create_collection(self._name, schema, **kwargs)
                self._schema = schema
                self._consistency_level = consistency_level
            else:
                raise SchemaNotReadyException(message=ExceptionsMessage.SchemaType)

        self._schema_dict = self._schema.to_dict()
        self._schema_dict["consistency_level"] = self._consistency_level

    def __repr__(self) -> str:
        _dict = {
            "name": self.name,
            "description": self.description,
            "schema": self._schema,
        }
        r = ["<Collection>:\n-------------\n"]
        s = "<{}>: {}\n"
        for k, v in _dict.items():
            r.append(s.format(k, v))
        return "".join(r)
    
    __str__ = __repr__

    def _get_connection(self)->AsyncGrpcHandler:
        handler = connections._fetch_handler(self._using)
        if not isinstance(handler, AsyncGrpcHandler):
            handler.__class__ = AsyncGrpcHandler    # type: ignore
        return handler  # type: ignore
        
    @classmethod
    def construct_from_dataframe(cls, name: str, dataframe: pd.DataFrame, **kwargs):
        if not isinstance(dataframe, pd.DataFrame):
            raise SchemaNotReadyException(message=ExceptionsMessage.DataFrameType)
        primary_field = kwargs.pop("primary_field", None)
        if primary_field is None:
            raise SchemaNotReadyException(message=ExceptionsMessage.NoPrimaryKey)
        pk_index = -1
        for i, field in enumerate(dataframe):
            if field == primary_field:
                pk_index = i
        if pk_index == -1:
            raise SchemaNotReadyException(message=ExceptionsMessage.PrimaryKeyNotExist)
        if "auto_id" in kwargs and not isinstance(kwargs.get("auto_id", None), bool):
            raise AutoIDException(message=ExceptionsMessage.AutoIDType)
        auto_id = kwargs.pop("auto_id", False)
        if auto_id:
            if dataframe[primary_field].isnull().all():
                dataframe = dataframe.drop(primary_field, axis=1)
            else:
                raise SchemaNotReadyException(message=ExceptionsMessage.AutoIDWithData)

        using = kwargs.get("using", Config.MILVUS_CONN_ALIAS)
        conn = _get_connection(using)
        if conn.has_collection(name, **kwargs):     # type: ignore
            resp = conn.describe_collection(name, **kwargs)
            server_schema = CollectionSchema.construct_from_dict(resp)
            schema = server_schema
        else:
            fields_schema = construct_fields_from_dataframe(dataframe)
            if auto_id:
                fields_schema.insert(
                    pk_index,
                    FieldSchema(
                        name=primary_field,
                        dtype=DataType.INT64,
                        is_primary=True,
                        auto_id=True,
                        **kwargs,
                    ),
                )

            for field in fields_schema:
                if auto_id is False and field.name == primary_field:
                    field.is_primary = True
                    field.auto_id = False
                if field.dtype == DataType.VARCHAR:
                    field.params[Config.MaxVarCharLengthKey] = int(Config.MaxVarCharLength)
            schema = CollectionSchema(fields=fields_schema)

        check_schema(schema)
        collection = cls(name, schema, **kwargs)
        res = collection.insert(data=dataframe) # type: ignore
        return collection, res

    @property
    def schema(self) -> CollectionSchema:
        """CollectionSchema: schema of the collection."""
        return self._schema

    @overload
    def get_aliases(self)->list: ...
    @overload
    async def get_aliases(self, _async=True)->list: ...

    def get_aliases(self, _async=False, **kwargs):
        """List[str]: all the aliases of the collection."""
        conn = self._get_connection()
        kwargs['_async'] = _async
        if _async:
            async def wrapper(kw):
                resp = await conn.describe_collection(self._name, **kw)
                return resp["aliases"]
            return wrapper(kwargs)
        else:
            resp = conn.describe_collection(self._name, **kwargs)
            return resp["aliases"]
    
    @property
    def aliases(self)->list:
        return self.get_aliases()
    
    @property
    def description(self) -> str:
        """str: a text description of the collection."""
        return self._schema.description

    @property
    def name(self) -> str:
        """str: the name of the collection."""
        return self._name

    @overload
    def check_is_empty(self)-> bool:...
    @overload
    async def check_is_empty(self, _async=True)->bool: ...
    
    def check_is_empty(self, _async=False):
        if _async:
            async def wrapper(self):
                return await self.get_num_entities(_async=True) == 0
            return wrapper(self)
        else:
            return self.num_entities == 0

    @property
    def is_empty(self) -> bool:
        """bool: whether the collection is empty or not."""
        return self.num_entities == 0

    @overload
    def get_num_shards(self, **kwargs)->int:...
    @overload
    async def get_num_shards(self, _async=True, **kwargs)->int:...
    
    def get_num_shards(self, _async=False, **kwargs):
        """int: number of shards used by the collection."""
        if _async:
            async def wrapper(**kwargs):
                resp = await self.describe(_async=True, **kwargs)
                return resp.get("num_shards")   # type: ignore
            return wrapper(**kwargs)
        else:
            if self._num_shards is None:
                self._num_shards = self.describe(timeout=kwargs.get("timeout")).get("num_shards")   # type: ignore
            return self._num_shards
    
    @property
    def num_shards(self) -> int:
        return self.get_num_shards()

    @overload
    def get_num_entities(self, **kwargs)->int:...
    @overload
    async def get_num_entities(self, _async=True, **kwargs)->int:...
    
    def get_num_entities(self, _async=False, **kwargs):
        """int: The number of entities in the collection, not real time."""
        conn = self._get_connection()
        if _async:
            async def wrapper(**kwargs):
                kwargs['_async'] = True
                stats = await conn.get_collection_stats(collection_name=self._name, **kwargs)
                result = {stat.key: stat.value for stat in stats}
                result["row_count"] = int(result["row_count"])
                return result["row_count"]
            return wrapper(**kwargs)
        else:
            stats = conn.get_collection_stats(collection_name=self._name, **kwargs)
            result = {stat.key: stat.value for stat in stats}
            result["row_count"] = int(result["row_count"])
            return result["row_count"]
    
    @property
    def num_entities(self, **kwargs) -> int:
        """int: The number of entities in the collection, not real time.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_num_entities", schema)
            >>> collection.num_entities
            0
            >>> collection.insert([[1, 2], [[1.0, 2.0], [3.0, 4.0]]])
            >>> collection.num_entities
            0
            >>> collection.flush()
            >>> collection.num_entities
            2
        """
        return self.get_num_entities(**kwargs)

    @property
    def primary_field(self) -> FieldSchema:
        """FieldSchema: the primary field of the collection."""
        return self._schema.primary_field   # type: ignore

    @overload
    def flush(self, timeout: Optional[float]=None, **kwargs): ...
    @overload
    async def flush(self, timeout: Optional[float]=None, _async=True, **kwargs): ...
    
    def flush(self, timeout: Optional[float] = None, _async=False, **kwargs):
        """Seal all segments in the collection. Inserts after flushing will be written into
            new segments. Only sealed segments can be indexed.

        Args:
            timeout (float): an optional duration of time in seconds to allow for the RPCs.
                If timeout is not set, the client keeps waiting until the server
                responds or an error occurs.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> fields = [
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=128)
            ... ]
            >>> schema = CollectionSchema(fields=fields)
            >>> collection = Collection(name="test_collection_flush", schema=schema)
            >>> collection.insert([[1, 2], [[1.0, 2.0], [3.0, 4.0]]])
            >>> collection.flush()
            >>> collection.num_entities
            2
        """
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, timeout, **kwargs):
                kwargs['_async'] = True
                await conn.flush([self.name], timeout=timeout, **kwargs)
            return wrapper(conn, timeout, **kwargs)
        else:
            conn.flush([self.name], timeout=timeout, **kwargs)

    @overload
    def drop_index(self, field_name: str, timeout: Optional[float] = None, **kwargs): ...
    @overload
    async def drop_index(self, field_name: str, timeout: Optional[float] = None, _async=True, **kwargs): ...
        
    def drop(self, timeout: Optional[float] = None, _async=False, **kwargs):
        """Drops the collection. The same as `utility.drop_collection()`

        Args:
            timeout (float, optional): an optional duration of time in seconds to allow
                for the RPCs. If timeout is not set, the client keeps waiting until the
                server responds or an error occurs.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_drop", schema)
            >>> utility.has_collection("test_collection_drop")
            True
            >>> collection.drop()
            >>> utility.has_collection("test_collection_drop")
            False
        """
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, timeout, **kwargs):
                kwargs['_async'] = True
                await conn.drop_collection(self._name, timeout=timeout, **kwargs)
            return wrapper(conn, timeout, **kwargs)
        else:
            conn.drop_collection(self._name, timeout=timeout, **kwargs)

    @overload
    def set_properties(self, properties: dict, timeout: Optional[float] = None, **kwargs): ...
    @overload
    async def set_properties(self, properties: dict, timeout: Optional[float] = None, _async=True, **kwargs): ...
    
    def set_properties(self, properties: dict, timeout: Optional[float] = None, _async=False, **kwargs):
        """Set properties for the collection

        Args:
            properties (``dict``): collection properties.
                 only support collection TTL with key `collection.ttl.seconds`
            timeout (float, optional): an optional duration of time in seconds to allow
                for the RPCs. If timeout is not set, the client keeps waiting until the
                server responds or an error occurs.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> fields = [
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=128)
            ... ]
            >>> schema = CollectionSchema(fields=fields)
            >>> collection = Collection("test_set_properties", schema)
            >>> collection.set_properties({"collection.ttl.seconds": 60})
        """
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, properties, timeout, **kwargs):
                kwargs['_async'] = True
                await conn.alter_collection(
                    self.name,
                    properties=properties,
                    timeout=timeout,
                    **kwargs,
                )
            return wrapper(conn, properties, timeout, **kwargs)
        else:
            conn.alter_collection(
                self.name,
                properties=properties,
                timeout=timeout,
                **kwargs,
            )

    @overload
    def load(self, partition_names: Optional[list] = None, replica_number: int = 1, timeout: Optional[float] = None, sync=True, **kwargs): ...
    @overload
    async def load(self, partition_names: Optional[list] = None, replica_number: int = 1, timeout: Optional[float] = None, sync=True, _async=True, **kwargs): ...
    
    def load(
        self,
        partition_names: Optional[list] = None,
        replica_number: int = 1,
        timeout: Optional[float] = None,
        sync=True,
        _async=False, 
        **kwargs,
    ):
        """Load the data into memory.

        Args:
            partition_names (``List[str]``): The specified partitions to load.
            replica_number (``int``, optional): The replica number to load, defaults to 1.
            timeout (float, optional): an optional duration of time in seconds to allow
                for the RPCs. If timeout is not set, the client keeps waiting until the
                server responds or an error occurs.
            **kwargs (``dict``, optional):

                * *_async*(``bool``)
                    Indicate if invoke asynchronously.

                * *_refresh*(``bool``)
                    Whether to renew the segment list of this collection before loading
                * *_resource_groups(``List[str]``)
                    Specify resource groups which can be used during loading.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_load", schema)
            >>> collection.insert([[1, 2], [[1.0, 2.0], [3.0, 4.0]]])
            >>> index_param = {"index_type": "FLAT", "metric_type": "L2", "params": {}}
            >>> collection.create_index("films", index_param)
            >>> collection.load()
        """
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, partition_names, replica_number, timeout, sync, **kwargs):
                kwargs['_async'] = True
                if partition_names is not None:
                    await conn.load_partitions(
                        collection_name=self._name,
                        partition_names=partition_names,
                        replica_number=replica_number,
                        timeout=timeout,
                        sync=sync,
                        **kwargs,
                    )
                else:
                    await conn.load_collection(
                        collection_name=self._name,
                        replica_number=replica_number,
                        timeout=timeout,
                        **kwargs,
                    )
            return wrapper(conn, partition_names, replica_number, timeout, sync, **kwargs)
        else:
            if partition_names is not None:
                conn.load_partitions(
                    collection_name=self._name,
                    partition_names=partition_names,
                    replica_number=replica_number,
                    timeout=timeout,
                    sync=sync,
                    **kwargs,
                )
            else:
                conn.load_collection(
                    collection_name=self._name,
                    replica_number=replica_number,
                    timeout=timeout,
                    **kwargs,
                )

    @overload
    def release(self, timeout: Optional[float] = None, **kwargs): ...
    @overload
    async def release(self, timeout: Optional[float] = None, _async=True, **kwargs): ...
    
    def release(self, timeout: Optional[float] = None, _async=False, **kwargs):
        """Releases the collection data from memory.

        Args:
            timeout (float, optional): an optional duration of time in seconds to allow
                for the RPCs. If timeout is not set, the client keeps waiting until the
                server responds or an error occurs.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_release", schema)
            >>> collection.insert([[1, 2], [[1.0, 2.0], [3.0, 4.0]]])
            >>> index_param = {"index_type": "FLAT", "metric_type": "L2", "params": {}}
            >>> collection.create_index("films", index_param)
            >>> collection.load()
            >>> collection.release()
        """
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, timeout, **kwargs):
                kwargs['_async'] = True
                await conn.release_collection(self._name, timeout=timeout, **kwargs)
            return wrapper(conn, timeout, **kwargs)
        else:
            conn.release_collection(self._name, timeout=timeout, **kwargs)

    @overload
    def insert(self, data: List, partition_name: Optional[str] = None, timeout: Optional[float] = None, **kwargs) -> MutationResult: ...
    @overload
    async def insert(self, data: List, partition_name: Optional[str] = None, timeout: Optional[float] = None, _async=True, **kwargs) -> MutationResult: ...
    
    def insert(
        self,
        data: Union[List, pd.DataFrame, Dict],
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        _async=False,
        **kwargs,
    ):
        """Insert data into the collection.

        Args:
            data (``list/tuple/pandas.DataFrame``): The specified data to insert
            partition_name (``str``): The partition name which the data will be inserted to,
                if partition name is not passed, then the data will be inserted
                to default partition
            timeout (float, optional): an optional duration of time in seconds to allow
                for the RPCs. If timeout is not set, the client keeps waiting until the
                server responds or an error occurs.
        Returns:
            MutationResult: contains 2 properties `insert_count`, and, `primary_keys`
                `insert_count`: how may entities have been inserted into Milvus,
                `primary_keys`: list of primary keys of the inserted entities
        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> import random
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_insert", schema)
            >>> data = [
            ...     [random.randint(1, 100) for _ in range(10)],
            ...     [[random.random() for _ in range(2)] for _ in range(10)],
            ... ]
            >>> res = collection.insert(data)
            >>> res.insert_count
            10
        """
        if data is None:
            return MutationResult(data)
        row_based = check_is_row_based(data)
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, data, partition_name, timeout, row_based, **kwargs):
                kwargs['_async'] = True
                if not row_based:
                    check_insert_schema(self._schema, data)
                    entities = Prepare.prepare_data(data, self._schema)
                    res = await conn.batch_insert(
                        self._name,
                        entities,
                        partition_name,
                        timeout=timeout,
                        schema=self._schema_dict,
                        **kwargs,
                    )
                else:
                    res = await conn.insert_rows(
                        collection_name=self._name,
                        entities=data,
                        partition_name=partition_name,
                        timeout=timeout,
                        schema=self._schema_dict,
                        **kwargs,
                    )
                return res
            return wrapper(conn, data, partition_name, timeout, row_based, **kwargs)
        else:
            if not row_based:
                check_insert_schema(self._schema, data) # type: ignore
                entities = Prepare.prepare_data(data, self._schema)  # type: ignore
                res = conn.batch_insert(
                    self._name,
                    entities,
                    partition_name,
                    timeout=timeout,
                    schema=self._schema_dict,
                    **kwargs,
                )
            else:
                res = conn.insert_rows(
                    collection_name=self._name,
                    entities=data,
                    partition_name=partition_name,
                    timeout=timeout,
                    schema=self._schema_dict,
                    **kwargs,
                )        
            return res

    @overload
    def delete(self, expr: str, partition_name: Optional[str] = None, timeout: Optional[float] = None, **kwargs) -> MutationResult: ...
    @overload
    async def delete(self, expr: str, partition_name: Optional[str] = None, timeout: Optional[float] = None, _async=True, **kwargs) -> MutationResult: ...
    
    def delete(
        self,
        expr: str,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        _async=False,
        **kwargs,
    ):
        """Delete entities with an expression condition.

        Args:
            expr (``str``): The specified data to insert.
            partition_names (``List[str]``): Name of partitions to delete entities.
            timeout (float, optional): an optional duration of time in seconds to allow
                for the RPCs. If timeout is not set, the client keeps waiting until the
                server responds or an error occurs.
            **kwargs (``dict``): Optional search params

                * *consistency_level* (``str/int``, optional)
                    Which consistency level to use when searching in the collection.

                    Options of consistency level: Strong, Bounded, Eventually, Session, Customized.

                    Note: this parameter overwrites the same one specified when creating collection,
                    if no consistency level was specified, search will use the
                    consistency level when you create the collection.

        Returns:
            MutationResult:
                contains `delete_count` properties represents how many entities might be deleted.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> import random
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("film_date", DataType.INT64),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2),
            ... ])
            >>> collection = Collection("test_collection_delete", schema)
            >>> # insert
            >>> data = [
            ...     [i for i in range(10)],
            ...     [i + 2000 for i in range(10)],
            ...     [[random.random() for _ in range(2)] for _ in range(10)],
            ... ]
            >>> collection.insert(data)
            >>> res = collection.delete("film_id in [ 0, 1 ]")
            >>> print(f"- Deleted entities: {res}")
            - Delete results: [0, 1]
        """

        conn = self._get_connection()
        if _async:
            async def wrapper(conn, expr, partition_name, timeout, **kwargs):
                kwargs['_async'] = True
                res = await conn.delete(
                    self._name, expr, partition_name, timeout=timeout, **kwargs
                )
                return res
                # return MutationResult(res)
            return wrapper(conn, expr, partition_name, timeout, **kwargs)
        else:
            res = conn.delete(self._name, expr, partition_name, timeout=timeout, **kwargs)
            # return MutationResult(res)
            return res

    @overload
    def upsert(self, data: List, partition_name: Optional[str] = None, timeout: Optional[float] = None, **kwargs) -> MutationResult: ...
    @overload
    async def upsert(self, data: List, partition_name: Optional[str] = None, timeout: Optional[float] = None, _async=True, **kwargs) -> MutationResult: ...
    
    def upsert(
        self,
        data: Union[List, pd.DataFrame, Dict],
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        _async=False,
        **kwargs,
    ):
        """Upsert data into the collection.

        Args:
            data (``list/tuple/pandas.DataFrame``): The specified data to upsert
            partition_name (``str``): The partition name which the data will be upserted at,
                if partition name is not passed, then the data will be upserted
                in default partition
            timeout (float, optional): an optional duration of time in seconds to allow
                for the RPCs. If timeout is not set, the client keeps waiting until the
                server responds or an error occurs.
        Returns:
            MutationResult: contains 2 properties `upsert_count`, and, `primary_keys`
                `upsert_count`: how may entities have been upserted at Milvus,
                `primary_keys`: list of primary keys of the upserted entities
        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> import random
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_upsert", schema)
            >>> data = [
            ...     [random.randint(1, 100) for _ in range(10)],
            ...     [[random.random() for _ in range(2)] for _ in range(10)],
            ... ]
            >>> res = collection.upsert(data)
            >>> res.upsert_count
            10
        """
        if data is None:
            r = MutationResult(data)
            if _async:
                async def wrapper(r):   # type: ignore
                    return r
                return wrapper(r)
            else:
                return r

        row_based = check_is_row_based(data)
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, data, partition_name, timeout, row_based, **kwargs):
                kwargs['_async'] = True
                if not row_based:
                    check_upsert_schema(self._schema, data)
                    entities = Prepare.prepare_data(data, self._schema, is_insert=False)
                    res = await conn.upsert(
                        self._name,
                        entities,
                        partition_name,
                        timeout=timeout,
                        schema=self._schema_dict,
                        **kwargs,
                    )
                else:
                    res = await conn.upsert_rows(
                        self._name,
                        data,
                        partition_name,
                        timeout=timeout,
                        schema=self._schema_dict,
                        **kwargs,
                    )
                return MutationFuture(res)  # cannot return `MutationResult(res)`
            return wrapper(conn, data, partition_name, timeout, row_based, **kwargs)
        else:
            if not row_based:
                check_upsert_schema(self._schema, data)  # type: ignore
                entities = Prepare.prepare_data(data, self._schema, is_insert=False)   # type: ignore
                res = conn.upsert(
                    self._name,
                    entities,
                    partition_name,
                    timeout=timeout,
                    schema=self._schema_dict,
                    **kwargs,
                )
            else:
                res = conn.upsert_rows(
                    self._name,
                    data,
                    partition_name,
                    timeout=timeout,
                    schema=self._schema_dict,
                    **kwargs,
                )
            return res # Will raise unknown error if return `MutationResult(res)``

    @overload
    def search(self, 
               data: List, 
               anns_field: str, 
               param: Dict, 
               limit: int, 
               expr:Optional[str]=None, 
               partition_names: Optional[List[str]]=None,
               output_fields: Optional[List[str]]=None,
               timeout: Optional[float]=None,
               round_decimal: int=-1, 
               **kwargs) -> SearchResult: ...
    
    @overload
    async def search(self, 
               data: List, 
               anns_field: str, 
               param: Dict, 
               limit: int, 
               expr:Optional[str]=None, 
               partition_names: Optional[List[str]]=None,
               output_fields: Optional[List[str]]=None,
               timeout: Optional[float]=None,
               round_decimal: int=-1,
               _async:bool=True, 
               **kwargs) -> SearchResult: ...
    
    def search(
        self,
        data: List,
        anns_field: str,
        param: Dict,
        limit: int,
        expr: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        round_decimal: int = -1,
        _async=False,
        **kwargs,
    ):
        """Conducts a vector similarity search with an optional boolean expression as filter.

        Args:
            data (``List[List[float]]``): The vectors of search data.
                the length of data is number of query (nq),
                and the dim of every vector in data must be equal to the vector field of collection.
            anns_field (``str``): The name of the vector field used to search of collection.
            param (``dict[str, Any]``):
                The parameters of search. The followings are valid keys of param.
                * *metric_type* (``str``)
                    similar metricy types, the value must be of type str.
                * *offset* (``int``, optional)
                    offset for pagination.
                * *params of index: *nprobe*, *ef*, *search_k*, etc
                    Corresponding search params for a certain index.
                example for param::

                    {
                        "metric_type": "L2",
                        "offset": 10,
                        "params": {"nprobe": 12},
                    }

            limit (``int``): The max number of returned record, also known as `topk`.
            expr (``str``, Optional): The boolean expression used to filter attribute.

                example for expr::

                    "id_field >= 0", "id_field in [1, 2, 3, 4]"

            partition_names (``List[str]``, optional): The names of partitions to search on.
            output_fields (``List[str]``, optional):
                The name of fields to return in the search result.  Can only get scalar fields.
            round_decimal (``int``, optional):
                The specified number of decimal places of returned distance.
                Defaults to -1 means no round to returned distance.
            timeout (``float``, optional): A duration of time in seconds to allow for the RPC.
                If timeout is set to None, the client keeps waiting until the server
                responds or an error occurs.
            **kwargs (``dict``): Optional search params

                *  *_async* (``bool``, optional)
                    Indicate if invoke asynchronously.
                    Returns a SearchFuture if True, else returns results from server directly.

                * *_callback* (``function``, optional)
                    The callback function which is invoked after server response successfully.
                    It functions only if _async is set to True.

                * *offset* (``int``, optional)
                    offset for pagination.

                * *consistency_level* (``str/int``, optional)
                    Which consistency level to use when searching in the collection.

                    Options of consistency level: Strong, Bounded, Eventually, Session, Customized.

                    Note: this parameter overwrites the same one specified when creating collection,
                    if no consistency level was specified, search will use the
                    consistency level when you create the collection.

                * *guarantee_timestamp* (``int``, optional)
                    Instructs Milvus to see all operations performed before this timestamp.
                    By default Milvus will search all operations performed to date.

                    Note: only valid in Customized consistency level.

                * *graceful_time* (``int``, optional)
                    Search will use the (current_timestamp - the graceful_time) as the
                    `guarantee_timestamp`. By default with 5s.

                    Note: only valid in Bounded consistency level

        Returns:
            SearchResult:
                Returns ``SearchResult`` if `_async` is False , otherwise ``SearchFuture``

        .. _Metric type documentations:
            https://milvus.io/docs/v2.2.x/metric.md
        .. _Index documentations:
            https://milvus.io/docs/v2.2.x/index.md
        .. _How guarantee ts works:
            https://github.com/milvus-io/milvus/blob/master/docs/developer_guides/how-guarantee-ts-works.md

        Raises:
            MilvusException: If anything goes wrong

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> import random
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_search", schema)
            >>> # insert
            >>> data = [
            ...     [i for i in range(10)],
            ...     [[random.random() for _ in range(2)] for _ in range(10)],
            ... ]
            >>> collection.insert(data)
            >>> index_param = {"index_type": "FLAT", "metric_type": "L2", "params": {}}
            >>> collection.create_index("films", index_param)
            >>> collection.load()
            >>> # search
            >>> search_param = {
            ...     "data": [[1.0, 1.0]],
            ...     "anns_field": "films",
            ...     "param": {"metric_type": "L2", "offset": 1},
            ...     "limit": 2,
            ...     "expr": "film_id > 0",
            ... }
            >>> res = collection.search(**search_param)
            >>> assert len(res) == 1
            >>> hits = res[0]
            >>> assert len(hits) == 2
            >>> print(f"- Total hits: {len(hits)}, hits ids: {hits.ids} ")
            - Total hits: 2, hits ids: [8, 5]
            >>> print(f"- Top1 hit id: {hits[0].id}, score: {hits[0].score} ")
            - Top1 hit id: 8, score: 0.10143111646175385
        """
        if expr is not None and not isinstance(expr, str):
            raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(expr))

        if isinstance(data, list) and len(data) == 0:
            resp = SearchResult(schema_pb2.SearchResultData())
            if _async:
                async def wrapper(r):    # type: ignore
                    return r
                return wrapper(resp)
            else:
                return resp

        conn = self._get_connection()
        if _async:
            kwargs['_async'] = True
            async def wrapper(conn):
                return await conn.search(
                    self._name,
                    data,
                    anns_field,
                    param,
                    limit,
                    expr,
                    partition_names,
                    output_fields,
                    round_decimal,
                    timeout=timeout,
                    schema=self._schema_dict,
                    **kwargs,
                )
            return wrapper(conn)
        else:
            resp = conn.search(
                self._name,
                data,
                anns_field,
                param,
                limit,
                expr,
                partition_names,
                output_fields,
                round_decimal,
                timeout=timeout,
                schema=self._schema_dict,
                **kwargs,
            )
            return resp

    def search_iterator(
        self,
        data: List,
        anns_field: str,
        param: Dict,
        batch_size: Optional[int] = 1000,
        limit: Optional[int] = UNLIMITED,
        expr: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        round_decimal: int = -1,
        **kwargs,
    ):
        if expr is not None and not isinstance(expr, str):
            raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(expr))
        return SearchIterator(
            connection=self._get_connection(),   # type: ignore
            collection_name=self._name,
            data=data,
            ann_field=anns_field,
            param=param,
            batch_size=batch_size,
            limit=limit,
            expr=expr,
            partition_names=partition_names,
            output_fields=output_fields,
            timeout=timeout,
            round_decimal=round_decimal,
            schema=self._schema_dict,    # type: ignore
            **kwargs,
        )
        
    @overload
    def query(self, expr: str, output_fields: Optional[List[str]] = None, partition_names: Optional[List[str]] = None, timeout: Optional[float] = None, _async=False, **kwargs) -> List:...
    @overload
    async def query(self, expr: str, output_fields: Optional[List[str]] = None, partition_names: Optional[List[str]] = None, timeout: Optional[float] = None, _async=True, **kwargs) -> List:...
    
    def query(
        self,
        expr: str,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        _async=False,
        **kwargs,
    ):
        """Query with expressions

        Args:
            expr (``str``): The query expression.
            output_fields(``List[str]``): A list of field names to return. Defaults to None.
            partition_names: (``List[str]``, optional): A list of partition names to query in.
            timeout (``float``, optional): A duration of time in seconds to allow for the RPC.
                If timeout is set to None, the client keeps waiting until the server
                responds or an error occurs.
            **kwargs (``dict``, optional):

                * *consistency_level* (``str/int``, optional)
                    Which consistency level to use when searching in the collection.

                    Options of consistency level: Strong, Bounded, Eventually, Session, Customized.

                    Note: this parameter overwrites the same one specified when creating collection,
                    if no consistency level was specified, search will use the
                    consistency level when you create the collection.


                * *guarantee_timestamp* (``int``, optional)
                    Instructs Milvus to see all operations performed before this timestamp.
                    By default Milvus will search all operations performed to date.

                    Note: only valid in Customized consistency level.

                * *graceful_time* (``int``, optional)
                    Search will use the (current_timestamp - the graceful_time) as the
                    `guarantee_timestamp`. By default with 5s.

                    Note: only valid in Bounded consistency level

                * *offset* (``int``)
                    Combined with limit to enable pagination

                * *limit* (``int``)
                    Combined with limit to enable pagination

        Returns:
            List, contains all results

        Raises:
            MilvusException: If anything goes wrong

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> import random
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("film_date", DataType.INT64),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_query", schema)
            >>> # insert
            >>> data = [
            ...     [i for i in range(10)],
            ...     [i + 2000 for i in range(10)],
            ...     [[random.random() for _ in range(2)] for _ in range(10)],
            ... ]
            >>> collection.insert(data)
            >>> index_param = {"index_type": "FLAT", "metric_type": "L2", "params": {}}
            >>> collection.create_index("films", index_param)
            >>> collection.load()
            >>> # query
            >>> expr = "film_id <= 1"
            >>> res = collection.query(expr, output_fields=["film_date"], offset=1, limit=1)
            >>> assert len(res) == 1
            >>> print(f"- Query results: {res}")
            - Query results: [{'film_id': 1, 'film_date': 2001}]
        """
        if not isinstance(expr, str):
            raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(expr))
        
        conn = self._get_connection()
        if _async:
            kwargs['_async'] = True
            async def wrapper(conn):
                return await conn.query(
                    self._name,
                    expr,
                    output_fields,
                    partition_names,
                    timeout=timeout,
                    schema=self._schema_dict,
                    **kwargs,
                )
            return wrapper(conn)
        else:
            return conn.query(
                self._name,
                expr,
                output_fields,
                partition_names,
                timeout=timeout,
                schema=self._schema_dict,
                **kwargs,
            )

    def query_iterator(
        self,
        batch_size: Optional[int] = 1000,
        limit: Optional[int] = UNLIMITED,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if expr is not None and not isinstance(expr, str):
            raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(expr))
        return QueryIterator(
            connection=self._get_connection(),   # type: ignore
            collection_name=self._name,
            batch_size=batch_size,
            limit=limit,
            expr=expr,
            output_fields=output_fields,
            partition_names=partition_names,
            schema=self._schema_dict,    # type: ignore
            timeout=timeout,
            **kwargs,
        )

    @overload
    def partitions(self, **kwargs) -> List[Partition]:...
    @overload
    async def partitions(self, _async=True, **kwargs) -> List[Partition]:...

    def partitions(self, _async=False, **kwargs):
        """List[Partition]: List of Partition object.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_partitions", schema)
            >>> collection.partitions
            [{"name": "_default", "description": "", "num_entities": 0}]
        """
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, kwargs):
                partition_strs = await conn.list_partitions(self._name, _async=True, **kwargs)
                partitions = []
                for partition in partition_strs:
                    partitions.append(Partition(self, partition, construct_only=True))
                return partitions
            return wrapper(conn, kwargs)
        else:
            partition_strs = conn.list_partitions(self._name, **kwargs)
            partitions = []
            for partition in partition_strs:
                partitions.append(Partition(self, partition, construct_only=True))
            return partitions

    @overload
    def partition(self, partition_name: str, **kwargs) -> Partition|None:...
    @overload
    async def partition(self, partition_name: str, _async=True, **kwargs) -> Partition|None:...

    def partition(self, partition_name: str, _async=False, **kwargs):
        """Get the existing partition object according to name. Return None if not existed.

        Args:
            partition_name (``str``): The name of the partition to get.

        Returns:
            Partition: Partition object corresponding to partition_name.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_partition", schema)
            >>> collection.partition("_default")
            {"name": "_default", "description": "", "num_entities": 0}
        """
        if _async:
            async def wrapper(conn, partition_name, kwargs):
                has_par = await conn.has_partition(self._name, partition_name, _async=True, **kwargs)
                if has_par is False:
                    return None
                return Partition(self, partition_name, construct_only=True)
            return wrapper(self._get_connection(), partition_name, kwargs)
        else:
            if not self.has_partition(partition_name, **kwargs):
                return None
            return Partition(self, partition_name, construct_only=True, **kwargs)

    @overload
    def create_partition(self, partition_name: str, description: str = "", **kwargs) -> Partition:...
    @overload
    async def create_partition(self, partition_name: str, description: str = "", _async=True, **kwargs) -> Partition:...

    def create_partition(self, partition_name: str, description: str = "", _async=False, **kwargs):
        """Create a new partition corresponding to name if not existed.
        Args:
            partition_name (``str``): The name of the partition to create.
            description (``str``, optional): The description of this partition.

        Returns:
            Partition: Partition object corresponding to partition_name.
        Raises:
            MilvusException: If anything goes wrong.
            
        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_create_partition", schema)
            >>> collection.create_partition("comedy", description="comedy films")
            {"name": "comedy", "collection_name": "test_create_partition", "description": ""}
            >>> collection.partition("comedy")
            {"name": "comedy", "collection_name": "test_create_partition", "description": ""}
        """
        if _async:
            async def wrapper(conn, partition_name, description, kwargs):
                has_par = await self.has_partition(partition_name, None, _async=True, **kwargs) # type: ignore
                if has_par is True:
                    raise PartitionAlreadyExistException(message=ExceptionsMessage.PartitionAlreadyExist)
                par = Partition(self, partition_name, description=description, construct_only=True, **kwargs) 
                await conn.create_partition(self.name, partition_name, _async=True, **kwargs)
                return par
            return wrapper(self._get_connection(), partition_name, description, kwargs)
        else:
            if self.has_partition(partition_name, **kwargs) is True:
                raise PartitionAlreadyExistException(message=ExceptionsMessage.PartitionAlreadyExist)
            return Partition(self, partition_name, description=description, **kwargs)

    @overload
    def has_partition(self, partition_name: str, timeout: Optional[float] = None, **kwargs) -> bool:...
    @overload
    async def has_partition(self, partition_name: str, timeout: Optional[float], _async=True, **kwargs) -> bool:...
    
    def has_partition(self, partition_name: str, timeout: Optional[float] = None, _async=False, **kwargs):
        """Checks if a specified partition exists.

        Args:
            partition_name (``str``): The name of the partition to check.
            timeout (``float``, optional): An optional duration of time in seconds to allow for
                the RPC. When timeout is set to None, client waits until server
                response or error occur.

        Returns:
            bool: True if exists, otherwise false.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_has_partition", schema)
            >>> collection.create_partition("comedy", description="comedy films")
            {"name": "comedy", "description": "comedy films", "num_entities": 0}
            >>> collection.has_partition("comedy")
            True
            >>> collection.has_partition("science_fiction")
            False
        """
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, partition_name, timeout, kwargs):
                kwargs['_async'] = True
                return await conn.has_partition(self._name, partition_name, timeout=timeout, **kwargs)
            return wrapper(conn, partition_name, timeout, kwargs)
        else:
            return conn.has_partition(self._name, partition_name, timeout=timeout, **kwargs)

    @overload
    def drop_partition(self, partition_name: str, timeout: Optional[float] = None, **kwargs):...
    @overload
    async def drop_partition(self, partition_name: str, timeout: Optional[float], _async=True, **kwargs):...
    
    def drop_partition(self, partition_name: str, timeout: Optional[float] = None, _async=False, **kwargs):
        """Drop the partition in this collection.

        Args:
            partition_name (``str``): The name of the partition to drop.
            timeout (``float``, optional): An optional duration of time in seconds to allow for
                the RPC. When timeout is set to None, client waits until server response
                or error occur.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_drop_partition", schema)
            >>> collection.create_partition("comedy", description="comedy films")
            {"name": "comedy", "description": "comedy films", "num_entities": 0}
            >>> collection.has_partition("comedy")
            True
            >>> collection.drop_partition("comedy")
            >>> collection.has_partition("comedy")
            False
        """
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, partition_name, timeout, kwargs):
                kwargs['_async'] = True
                return await conn.drop_partition(self._name, partition_name, timeout=timeout, **kwargs)
            return wrapper(conn, partition_name, timeout, kwargs)
        else:
            return conn.drop_partition(self._name, partition_name, timeout=timeout, **kwargs)

    @overload
    def indexes(self, **kwargs) -> list[Index]:...
    @overload
    async def indexes(self, _async=True, **kwargs) -> list[Index]:...
    
    def indexes(self, _async=False, **kwargs):
        """List[Index]: list of indexes of this collection.
        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_indexes", schema)
            >>> collection.indexes
            []
        """
        conn = self._get_connection()
        indexes = []
        if _async:
            async def wrapper(conn, kwargs):
                kwargs['_async'] = True
                tmp_index = await conn.list_indexes(self._name, **kwargs)
                for index in tmp_index:
                    if index is not None:
                        info_dict = {kv.key: kv.value for kv in index.params}
                        if info_dict.get("params", None):
                            info_dict["params"] = json.loads(info_dict["params"])

                        index_info = Index(
                            collection=self,
                            field_name=index.field_name,
                            index_params=info_dict,
                            index_name=index.index_name,
                            construct_only=True,
                        )
                        indexes.append(index_info)
                return indexes
            return wrapper(conn, kwargs)
        else:
            tmp_index = conn.list_indexes(self._name, **kwargs)
            for index in tmp_index:
                if index is not None:
                    info_dict = {kv.key: kv.value for kv in index.params}
                    if info_dict.get("params", None):
                        info_dict["params"] = json.loads(info_dict["params"])

                    index_info = Index(
                        collection=self,
                        field_name=index.field_name,
                        index_params=info_dict,
                        index_name=index.index_name,
                        construct_only=True,
                    )
                    indexes.append(index_info)
            return indexes

    @overload
    def index(self, **kwargs) -> Index:...
    @overload
    async def index(self, _async=True, **kwargs) -> Index:...
    
    def index(self, _async=False, **kwargs):
        """Get the index object of index name.
        Args:
            **kwargs (``dict``):
                * *index_name* (``str``)
                    The name of index. If no index is specified, the default index name is used.
        Returns:
            Index: Index object corresponding to index_name.
        Raises:
            IndexNotExistException: If the index doesn't exists.
        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_index", schema)
            >>> index = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
            >>> collection.create_index("films", index)
            Status(code=0, message='')
            >>> collection.indexes
            [<pymilvus.index.Index object at 0x7f4435587e20>]
            >>> collection.index()
            <pymilvus.index.Index object at 0x7f44355a1460>
        """
        copy_kwargs = copy.deepcopy(kwargs)
        index_name = copy_kwargs.pop("index_name", Config.IndexName)
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, index_name, kwargs):
                kwargs['_async'] = True
                tmp_index = await conn.describe_index(self._name, index_name, **kwargs)
                if tmp_index is not None:
                    field_name = tmp_index.pop("field_name", None)
                    index_name = tmp_index.pop("index_name", index_name)
                    return Index(self, field_name, tmp_index, construct_only=True, index_name=index_name)
                raise IndexNotExistException(message=ExceptionsMessage.IndexNotExist)
            return wrapper(conn, index_name, copy_kwargs)
        else:
            tmp_index = conn.describe_index(self._name, index_name, **copy_kwargs)
            if tmp_index is not None:
                field_name = tmp_index.pop("field_name", None)
                index_name = tmp_index.pop("index_name", index_name)
                return Index(self, field_name, tmp_index, construct_only=True, index_name=index_name)
            raise IndexNotExistException(message=ExceptionsMessage.IndexNotExist)

    @overload
    def create_index(self, field_name: str, index_params: Dict, timeout: Optional[float] = None, sync=True, **kwargs):...
    @overload
    async def create_index(self, field_name: str, index_params: Dict, timeout: Optional[float] = None, sync=True, _async=True, **kwargs):...

    def create_index(
        self,
        field_name: str,
        index_params: Optional[Dict] = None,
        timeout: Optional[float] = None,
        sync:bool=True,
        _async=False,
        **kwargs,
    ):
        """Creates index for a specified field, with a index name.

        Args:
            field_name (``str``): The name of the field to create index
            index_params (``dict``, optional): The parameters to index
                * *index_type* (``str``)
                    "index_type" as the key, example values: "FLAT", "IVF_FLAT", etc.

                * *metric_type* (``str``)
                    "metric_type" as the key, examples values: "L2", "IP", "JACCARD".

                * *params* (``dict``)
                    "params" as the key, corresponding index params.

            timeout (``float``, optional): An optional duration of time in seconds to allow
                for the RPC. When timeout is set to None, client waits until server
                response or error occur.
            index_name (``str``): The name of index which will be created, must be unique.
                If no index name is specified, the default index name will be used.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_create_index", schema)
            >>> index_params = {
            ...     "index_type": "IVF_FLAT",
            ...     "params": {"nlist": 128},
            ...     "metric_type": "L2"}
            >>> collection.create_index("films", index_params, index_name="idx")
            Status(code=0, message='')
        """
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, field_name, index_params, timeout, sync, kwargs):
                kwargs['_async'] = True
                return await conn.create_index(self._name, field_name, index_params, timeout=timeout, sync=sync, **kwargs)
            return wrapper(conn, field_name, index_params, timeout, sync, kwargs)
        else:
            return conn.create_index(self._name, field_name, index_params, timeout=timeout, sync=sync, **kwargs)

    @overload
    def has_index(self, timeout: Optional[float] = None, **kwargs) -> bool:...
    @overload
    async def has_index(self, timeout: Optional[float] = None, _async=True, **kwargs) -> bool:...

    def has_index(self, timeout: Optional[float] = None, _async=False, **kwargs):
        """Check whether a specified index exists.

        Args:
            timeout (``float``, optional): An optional duration of time in seconds to allow
                for the RPC. When timeout is set to None, client waits until server response
                or error occur.

            **kwargs (``dict``):
                * *index_name* (``str``)
                  The name of index. If no index is specified, the default index name will be used.

        Returns:
            bool: Whether the specified index exists.

        Examples:
            >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_has_index", schema)
            >>> index = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
            >>> collection.create_index("films", index)
            >>> collection.has_index()
            True
        """
        conn = self._get_connection()
        copy_kwargs = copy.deepcopy(kwargs)
        index_name = copy_kwargs.pop("index_name", Config.IndexName)
        if _async:
            async def wrapper(conn, timeout, index_name, kwargs):
                kwargs['_async'] = True
                tmp_index = await conn.describe_index(self._name, index_name, timeout=timeout, **kwargs)
                if tmp_index is None:
                    return False
                return True
            return wrapper(conn, timeout, index_name, copy_kwargs)
        else:
            if conn.describe_index(self._name, index_name, timeout=timeout, **copy_kwargs) is None:
                return False
            return True

    @overload
    def drop_index(self, timeout: Optional[float] = None, **kwargs):...
    @overload
    async def drop_index(self, timeout: Optional[float] = None, _async=True, **kwargs):...

    def drop_index(self, timeout: Optional[float] = None, _async=False, **kwargs):  # type: ignore
        """Drop index and its corresponding index files.
        Args:
            timeout (``float``, optional): An optional duration of time in seconds to allow
                for the RPC. When timeout is set to None, client waits until server response
                or error occur.

            **kwargs (``dict``):
                * *index_name* (``str``)
                  The name of index. If no index is specified, the default index name will be used.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus Collection, FieldSchema, CollectionSchema, DataType
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_has_index", schema)
            >>> index = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
            >>> collection.create_index("films", index)
            >>> collection.has_index()
            True
            >>> collection.drop_index()
            >>> collection.has_index()
            False
        """
        copy_kwargs = copy.deepcopy(kwargs)
        index_name = copy_kwargs.pop("index_name", Config.IndexName)
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, timeout, index_name, kwargs):
                kwargs['_async'] = True
                tmp_index = await conn.describe_index(self._name, index_name, timeout=timeout, **kwargs)
                if tmp_index is not None:
                    await conn.drop_index(
                        collection_name=self.name,
                        field_name=tmp_index["field_name"],
                        index_name=index_name,
                        timeout=timeout,
                        **kwargs,
                    )
            return wrapper(conn, timeout, index_name, copy_kwargs)
        else:
            tmp_index = conn.describe_index(self._name, index_name, timeout=timeout, **copy_kwargs)
            if tmp_index is not None:
                index = Index(
                    collection=self,
                    field_name=tmp_index["field_name"],
                    index_params=tmp_index,
                    construct_only=True,
                    index_name=index_name,
                )
                index.drop(timeout=timeout, **kwargs)

    @overload
    def compact(self, timeout: Optional[float] = None, **kwargs):...
    @overload
    async def compact(self, timeout: Optional[float] = None, _async=True, **kwargs):...
    
    def compact(self, timeout: Optional[float] = None, _async=False, **kwargs):
        """Compact merge the small segments in a collection
        Args:
            timeout (``float``, optional): An optional duration of time in seconds to allow
                for the RPC. When timeout is set to None, client waits until server response
                or error occur.
        Raises:
            MilvusException: If anything goes wrong.
        """
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, timeout, kwargs):
                kwargs['_async'] = True
                return await conn.compact(self._name, timeout=timeout, **kwargs)
            return wrapper(conn, timeout, kwargs)
        else:
            self.compaction_id = conn.compact(self._name, timeout=timeout, **kwargs)

    @overload
    def get_compaction_state(self, timeout: Optional[float] = None, **kwargs) -> CompactionState:...
    @overload
    async def get_compaction_state(self, timeout: Optional[float] = None, _async=True, **kwargs) -> CompactionState:...

    def get_compaction_state(self, timeout: Optional[float] = None, _async=True, **kwargs):
        """Get the current compaction state
        Args:
            timeout (``float``, optional): An optional duration of time in seconds to allow
                for the RPC. When timeout is set to None, client waits until server response
                or error occur.
        Raises:
            MilvusException: If anything goes wrong.
        """
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, timeout, kwargs):
                kwargs['_async'] = True
                return await conn.get_compaction_state(self.compaction_id, timeout=timeout, **kwargs)
            return wrapper(conn, timeout, kwargs)
        else:
            return conn.get_compaction_state(self.compaction_id, timeout=timeout, **kwargs)

    @overload
    def wait_for_compaction_completed(self, timeout: Optional[float] = None, **kwargs) -> CompactionState:...
    @overload
    async def wait_for_compaction_completed(self, timeout: Optional[float] = None, _async=True, **kwargs) -> CompactionState:...

    def wait_for_compaction_completed(self, timeout: Optional[float] = None, _async=False, **kwargs):
        """Block until the current collection's compaction completed
        Args:
            timeout (``float``, optional): An optional duration of time in seconds to allow
                for the RPC. When timeout is set to None, client waits until server response
                or error occur.
        Raises:
            MilvusException: If anything goes wrong.
        """
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, timeout, kwargs):
                kwargs['_async'] = True
                return await conn.wait_for_compaction_completed(self.compaction_id, timeout=timeout, **kwargs)
            return wrapper(conn, timeout, kwargs)
        else:
            return conn.wait_for_compaction_completed(self.compaction_id, timeout=timeout, **kwargs)

    @overload
    def get_compaction_plans(self, timeout: Optional[float] = None, **kwargs) -> CompactionPlans:...
    @overload
    async def get_compaction_plans(self, timeout: Optional[float] = None, _async=True, **kwargs) -> CompactionPlans:...
    
    def get_compaction_plans(self, timeout: Optional[float] = None, _async=False, **kwargs):
        """Get the current compaction plans

        Args:
            timeout (``float``, optional): An optional duration of time in seconds to allow
                for the RPC. When timeout is set to None, client waits until server response
                or error occur.
        Returns:
            CompactionPlans: All the plans' states of this compaction.
        """
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, timeout, kwargs):
                kwargs['_async'] = True
                return await conn.get_compaction_plans(self.compaction_id, timeout=timeout, **kwargs)
            return wrapper(conn, timeout, kwargs)
        else:
            return conn.get_compaction_plans(self.compaction_id, timeout=timeout, **kwargs)

    @overload
    def get_replicas(self, timeout: Optional[float] = None, **kwargs) -> Replica:...
    @overload
    async def get_replicas(self, timeout: Optional[float] = None, _async=True, **kwargs) -> Replica:...
    
    def get_replicas(self, timeout: Optional[float] = None, _async=False, **kwargs):
        """Get the current loaded replica information

        Args:
            timeout (``float``, optional): An optional duration of time in seconds to allow
                for the RPC. When timeout is set to None, client waits until server response
                or error occur.
        Returns:
            Replica: All the replica information.
        """
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, timeout):
                return await conn.get_replicas(self.name, timeout=timeout, _async=True, **kwargs)
            return wrapper(conn, timeout)
        else:
            return conn.get_replicas(self.name, timeout=timeout, **kwargs)

    @overload
    def describe(self, timeout: Optional[float] = None) -> CollectionSchema:...
    @overload
    async def describe(self, timeout: Optional[float] = None, _async=True) -> CollectionSchema:...
    
    def describe(self, timeout: Optional[float] = None, _async=False):
        conn = self._get_connection()
        if _async:
            async def wrapper(conn, timeout):
                return await conn.describe_collection(self.name, timeout=timeout, _async=True)
            return wrapper(conn, timeout)
        else:
            return conn.describe_collection(self.name, timeout=timeout)


__all__ = ["Collection"]
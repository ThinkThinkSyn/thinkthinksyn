if __name__ == "__main__":  # for debugging
    import os, sys
    _proj_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    sys.path.append(_proj_path)
    __package__ = "utils.common.type_utils"

import re
import json
import base64

from datetime import datetime, date
from attr import asdict as attr_asdict
from pydantic import TypeAdapter, BaseModel
from pydantic_core import PydanticSerializationError
from dataclasses import asdict as datacls_asdict
from typing_extensions import TypeForm
from types import GenericAlias, UnionType
from typing import (Any, Sequence, TypeAlias, overload, no_type_check, Iterable, 
                    Literal, TYPE_CHECKING, get_origin as tp_get_origin, Annotated, Union)
from typing import (
    _GenericAlias, # type: ignore
    _LiteralGenericAlias,  # type: ignore
    _UnionGenericAlias, # type: ignore
    _UnpackGenericAlias,  # type: ignore
    _CallableGenericAlias,  # type: ignore
)
from typing_extensions import TypeAliasType

from .type_helpers import BasicType, BaseModelType

SerializableType: TypeAlias = BasicType | date | datetime | BaseModelType
"""Serializable type, including basic types and pydantic BaseModel"""

__serializable_types__: dict[str, bool] = dict()

_iso_time_regex = re.compile(r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}.\d{3,}$")
_iso_date_regex = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def is_serializable(val: Any) -> bool:
    """
    Check if the value is serializable. Any following objects:
        - has `__serialize__` method (highest priority)
        - is basic type(int, float, str, bool, list, dict, tuple, set)
        - is pydantic BaseModel
        - has `__get_pydantic_core_schema__` method, and the schema has `serialization` method
            will be considered as serializable.
    """
    from .checking import check_type_is, _save_isinstance, _tidy_type
    from .type_helpers import get_cls_name
    
    if isinstance(val, (type, GenericAlias, TypeAliasType, _GenericAlias, _LiteralGenericAlias,     # type: ignore
                        _UnionGenericAlias, _UnpackGenericAlias, _CallableGenericAlias)):   # type: ignore
        val = _tidy_type(val)[0]    # type: ignore
    
    # check special types
    val_origin = tp_get_origin(val)
    if val_origin is Annotated:
        val = val.__args__[0]  # type: ignore
        val_origin = tp_get_origin(val)
    if val_origin is Literal:  # type: ignore
        # Literal is always serializable
        return True
    elif val_origin in (Union, UnionType):  # type: ignore
        return all(is_serializable(t) for t in val.__args__)  # type: ignore
    
    if hasattr(val, "__serialize__"):
        return True
    if not _save_isinstance(val, type) and not isinstance(val, GenericAlias):
        val_type = type(val)
    else:
        val_type = val
    
    val_type_name = get_cls_name(val_type, with_module_name=True)
    if val_type_name in __serializable_types__:
        return __serializable_types__[val_type_name]
    
    ok = check_type_is(val_type, SerializableType)
    if not ok:
        from .type_helpers import is_attrs_cls, is_dataclass
        if is_attrs_cls(val):
            for field in val.__attrs_attrs__:   # type: ignore
                if not is_serializable(field.type):
                    return False
            return True
        elif is_dataclass(val):
            for field in val.__dataclass_fields__.values():     # type: ignore
                if not is_serializable(field.type):
                    return False
            return True
        elif hasattr(val_type, "__get_pydantic_core_schema__"):
            try:
                adapter = get_pydantic_type_adapter(val_type)   # type: ignore
                schema = adapter.core_schema
                if 'serialization' in schema:
                    return True
                return False
            except Exception:
                pass
    else:
        if isinstance(val_type, GenericAlias):
            # check inner types of GenericAlias
            if hasattr(val_type, "__args__"):
                for inner_type in val_type.__args__:
                    if not is_serializable(inner_type):
                        return False
    __serializable_types__[val_type_name] = ok
    return ok

def serialize(
    val: SerializableType, 
    do_not_serialize_str: bool=False, 
    bytes_to_b64: bool=True
) -> str:
    """
    Serialize the value to string.

    Args:
        val: the value to serialize
        do_not_serialize_str: if true, for string(e.g. 'abc'), it will not be serialized to '"abc"'.
        bytes_to_b64: if true, bytes will be converted to base64 string.

    Special cases:
        - you has defined `__serialize__` method in the value, it will have the highest priority,
          you should return a basic serializable type in the method.
        - __get_pydantic_core_schema__ is defined in the value, and serialization method is defined
         inside, it will use the schema to serialize the value.
        - for date/datetime, it will be serialized to ISO format string.
    """ 
    if isinstance(val, str):
        if not do_not_serialize_str:
            return val
        return f'"{val}"'
    
    if not is_serializable(val):
        raise TypeError(f"Type {type(val)} is not serializable.")
    
    if hasattr(val, "__serialize__"):
        val = val.__serialize__()  # type: ignore
        if not isinstance(val, str):
            return serialize(val)
        else:
            return val
    
    elif isinstance(val, bytes):
        if bytes_to_b64:
            return base64.b64encode(val).decode("utf-8")
        raise TypeError(f"Cannot serialize bytes to string if `bytes_to_b64` is False.")
    
    if hasattr(val, "__get_pydantic_core_schema__"):
        adapter = get_pydantic_type_adapter(type(val))  # type: ignore
        try:
            return adapter.dump_json(val).decode("utf-8")  # type: ignore
        except PydanticSerializationError:
            val = adapter.dump_python(val)  # type: ignore
            # dump to dict first, then use `recursive_dump_to_basic_types` to convert to basic types
    
    data = recursive_dump_to_basic_types(val, bytes_to_b64=bytes_to_b64)
    return json.dumps(data, ensure_ascii=False)


__pydantic_type_adapters__ = dict()
"""{type_name, adapter}"""

@overload
def get_pydantic_type_adapter[T](t: TypeForm[T]) -> TypeAdapter[T]: ...
@overload
def get_pydantic_type_adapter[T](t: type[T]) -> TypeAdapter[T]: ...
@overload
def get_pydantic_type_adapter[T: TypeAliasType](t: T) -> TypeAdapter[T]: ...
@overload
def get_pydantic_type_adapter(t: str) -> TypeAdapter: ...

def get_pydantic_type_adapter(t) -> TypeAdapter:
    """
    Get pydantic type adapter by type name or type.
    If name is give instead of type, make sure that the real type
    has been defined somewhere before getting adapter, otherwise
    error will be raised.

    Cache will be used to prevent multiple loading of the same type.
    """
    from .type_helpers import get_cls_name, get_module_name
    from .checking import _save_isinstance
    
    if _save_isinstance(t, TypeAliasType):
        type_name = f'{get_module_name(t)}{t.__name__}'
    else:
        type_name = t if isinstance(t, str) else get_cls_name(t, with_module_name=True)
    
    if type_name in __pydantic_type_adapters__:
        return __pydantic_type_adapters__[type_name]
    adapter = TypeAdapter(t)
    __pydantic_type_adapters__[type_name] = adapter
    return adapter

def _get_num_from_text(s: str, raise_error: bool = False):
    if s.startswith('0x'):
        try:
            return int(s, 16)
        except:
            ...
    elif s.startswith('0b'):
        try:
            return int(s, 2)
        except:
            ...
    elif s.startswith('0o'):
        try:
            return int(s, 8)
        except:
            ...
    try:
        return float(s)
    except:
        ...
    if raise_error:
        raise ValueError(f'Cannot convert `{s}` to number')
    return None

@overload
def deserialize[T](val: str, target_type: type[T]) -> T: ...    # not using `:SerializableType` to support some special serializable types
@overload
def deserialize[T](val: str, target_type: type[T], allow_none: Literal[False]=False) -> T: ...    # not using `:SerializableType` to support some special serializable types
@overload
def deserialize[T](val: str, target_type: type[T], allow_none: Literal[True]=True) -> T|None: ...    # not using `:SerializableType` to support some special serializable types
@overload
def deserialize(val: str, target_type: str) -> SerializableType: ...
@overload
def deserialize(val: str, target_type: str, allow_none: Literal[False]=False) -> SerializableType: ...
@overload
def deserialize(val: str, target_type: str, allow_none: Literal[True]=True) -> SerializableType|None: ...
@overload
def deserialize(val: str) -> SerializableType: ...

@no_type_check
def deserialize(val: str, target_type: type | str | None = None, allow_none: bool|None=None):  # type: ignore
    """
    Deserialize the string to the target type.
    If the deserialized value is not the target type, will try to convert it to the target type.

    Args:
        val: the string to deserialize
        target_type: The target type to convert the deserialized value to. If not given, will try to
                     The final return type will just be determined by `json.loads`.
                     You can also pass a type name by string, e.g. 'int', 'float', 'str', 'bool', 'list', 'dict', etc,
                     but not all types are supported.
        allow_none: when target_type is specified, but the value is None, it will still be returned.
                    This param has different default value for different overloads:
                        - when has target_type, default False
                        - when no target_type, default True
    If:
        - `__deserialize__` classmethod is defined in the target type, it will have the highest priority.
        - `__get_pydantic_core_schema__` is defined in the target type, pydantic's type adapter will be used
            for deserialization.
    """
    from .checking import _tidy_type, check_value_is, _save_issubclass, _save_isinstance
    origin_val = val
    
    if target_type:
        try:
            target_type = _tidy_type(target_type)[0]  # type: ignore
        except:
            ...
    
    if not _save_isinstance(val, str):
        if target_type:
            if check_value_is(val, target_type):
                return val
            raise TypeError(f"Cannot deserialize non-string type `{val}`({type(target_type)}) to {target_type}.")
        return val  # ignore when val is not str
        
    if target_type:
        if allow_none is None:
            allow_none = False
        if allow_none and val.lower().strip() in ("null", 'none'):
            return None
        if hasattr(target_type, "__deserialize__"):
            try:
                return target_type.__deserialize__(val)  # type: ignore
            except:
                pass
        if hasattr(target_type, "__get_pydantic_core_schema__"):
            adapter = get_pydantic_type_adapter(target_type)  # type: ignore
            try:
                return adapter.validate_json(val)  # type: ignore
            except Exception as e:
                if _save_issubclass(target_type, str) and target_type!=str:
                    return adapter.validate_python(val)
                else:
                    raise e
    else:
        if allow_none is None:
            allow_none = True
    
    try:
        if target_type == Any:
            target_type = None
        
        if target_type == bytes:
            return base64.b64decode(val.encode("utf-8"))  # type: ignore
        
        if target_type == str:
            if val.startswith('"') and val.endswith('"'):
                return val[1:-1]  # type: ignore
            return val  # type: ignore
        
        if target_type == bool:
            false_vals = ("false", "0", "no", "off")
            true_vals = ("true", "1", "yes", "on", 'ok')
            sim_val = val.lower().strip() 
            if sim_val in false_vals:
                return False
            elif sim_val in true_vals:
                return True
        
        if hasattr(target_type, "__origin__") and target_type.__origin__ is Literal:    # type: ignore
            if val in target_type.__args__: # type: ignore
                return val  # type: ignore
            raise ValueError(f"Value `{val}` is not a valid Literal for {target_type}.")
        
        if target_type:
            if _save_issubclass(target_type, datetime):
                if _save_isinstance(val, str):
                    if _iso_time_regex.match(val):
                        return datetime.fromisoformat(val)
                    raise ValueError(f"Cannot deserialize string `{origin_val}` to {target_type}.")
                if _save_isinstance(val, (int, float)):
                    return datetime.fromtimestamp(val)
                raise TypeError(f"Cannot deserialize non-string type `{val}`({type(val)}) to {target_type}.")
            
            elif _save_issubclass(target_type, date):
                if _save_isinstance(val, str):
                    if _iso_date_regex.match(val):
                        return date.fromisoformat(val)
                    raise ValueError(f"Cannot deserialize string `{origin_val}` to {target_type}.")
                if _save_isinstance(val, (int, float)):
                    return date.fromtimestamp(val)
                raise TypeError(f"Cannot deserialize non-string type `{val}`({type(val)}) to {target_type}.")
            
            elif _save_issubclass(target_type, (int, float)):
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                maybe_val = _get_num_from_text(val, raise_error=False)
                if maybe_val is not None:
                    val = maybe_val
                    if _save_issubclass(target_type, int):
                        val = int(val)
                    elif _save_issubclass(target_type, float):
                        val = float(val)
                    if target_type in (int, float):
                        return val  # can be returned directly
                    # wait for type adapter to do further validation
                
            adapter = get_pydantic_type_adapter(target_type)
            if _save_isinstance(val, str):
                return adapter.validate_json(val)  # type: ignore
            return adapter.validate_python(val)  # type: ignore
        else:
            if _save_isinstance(val, str):
                if _iso_time_regex.match(val):
                    return datetime.fromisoformat(val)
                if _iso_date_regex.match(val):
                    return date.fromisoformat(val)
            return json.loads(val)  # type: ignore
        
    except Exception as e:
        if target_type:
            raise TypeError(f"Cannot deserialize string `{origin_val}` to {target_type}.") from e
        raise TypeError(f"Cannot deserialize string `{origin_val}`.") from e

@overload
def recursive_dump_to_basic_types[T: BasicType](
    data: T, 
    ignore_err: bool=False, 
    include_non_dumpable: bool=True, 
    bytes_to_b64: bool=False,
    no_recursion_limit: bool=False,
    excepts: type|Sequence[type]|None=None,
) -> T: ...
@overload
def recursive_dump_to_basic_types(
    data: BaseModelType, 
    ignore_err: bool=False, 
    include_non_dumpable: bool=True, 
    bytes_to_b64: bool=False,
    no_recursion_limit: bool=False,
    excepts: type|Sequence[type]|None=None,
)->dict:...
@overload
def recursive_dump_to_basic_types(
    data: Any, 
    ignore_err: bool=False, 
    include_non_dumpable: bool=True, 
    bytes_to_b64: bool=False,
    no_recursion_limit: bool=False,
    excepts: type|Sequence[type]|None=None,
) -> BasicType: ...


_SHOULD_NOT_INCLUDE = object()
MAX_RECURSIVE_DEPTH = 512


def _internal_recursive_dump_to_basic_types(
    data: Any,
    _dumped_ids: set|None=None,
    ignore_err:bool=False,
    include_non_dumpable:bool=True,
    inner:bool=False,
    bytes_to_b64:bool=False,
    no_recursion_limit:bool=False,
    excepts: tuple[type, ...]|None=None,
    _recursive_depth:int=0,
):
    from .checking import check_value_is
    
    if not no_recursion_limit and _recursive_depth > MAX_RECURSIVE_DEPTH:
        return _SHOULD_NOT_INCLUDE
    
    def dump(data, _dumped_ids: set|None=None): # type: ignore
        try:
            if isinstance(data, (datetime, date)):
                return data.isoformat(), None
            if hasattr(data, "model_dump"):  # basemodel v2
                if TYPE_CHECKING:
                    data: BaseModel = data
                try:
                    return data.model_dump(), None
                except ValueError as e:
                    if 'Circular reference' in str(e):
                        data_dict = {}
                        for name, field in data.model_fields.items():
                            serialization_alias = field.serialization_alias or field.alias or name
                            data_attr = getattr(data, name)
                            if data_attr is _SHOULD_NOT_INCLUDE:
                                continue
                            data_attr = _internal_recursive_dump_to_basic_types(
                                data=getattr(data, name),
                                ignore_err=ignore_err,
                                include_non_dumpable=include_non_dumpable,
                                bytes_to_b64=bytes_to_b64,
                                no_recursion_limit=no_recursion_limit,
                                excepts=excepts,
                                _recursive_depth=_recursive_depth + 1,
                                _dumped_ids=_dumped_ids.copy() if _dumped_ids is not None else None,
                            )
                            if data_attr is not _SHOULD_NOT_INCLUDE:
                                data_dict[serialization_alias] = data_attr
                        return data_dict, None
                    raise e
            if hasattr(data, "dict") and callable(getattr(data, 'dict')):  # basemodel v1
                return data.dict(), None
            if hasattr(data, "to_list"):
                return data.to_list(), None
            if hasattr(data, "asdict"):  # attr class
                return data.asdict(), None
            if hasattr(data, "__dataclass_fields__"):  # dataclass
                return datacls_asdict(data), None
            if hasattr(data, "__attrs_attrs__"):
                return attr_asdict(data), None
            if hasattr(data, '__get_pydantic_core_schema__'):   # this must be put at the end, otherwise RecursiveError will be raised
                return get_pydantic_type_adapter(type(data)).dump_python(data), None
        except Exception as e:
            return data, e
        return data, TypeError(f"Cannot dump data `{data}` to basic types. Type of data: {type(data)}")

    if excepts:
        for t in excepts:
            # ignore dump for excepted types
            if check_value_is(data, t):
                return data 
    
    if not check_value_is(data, BasicType):
        data_id = id(data)
        if _dumped_ids is not None and data_id in _dumped_ids:
            return _SHOULD_NOT_INCLUDE
        if _dumped_ids is None:
            _dumped_ids_copy = set()
        else:
            _dumped_ids_copy = _dumped_ids.copy()
        _dumped_ids_copy.add(data_id)
        data, err = dump(data, _dumped_ids_copy)
        if err:
            if ignore_err:
                if include_non_dumpable:
                    return data
                elif inner:
                    return _SHOULD_NOT_INCLUDE
                else:
                    return data  # still return the non-dumpable data in outermost level
            raise err   # type: ignore
        # if `data` is dumped as dict, will keep checking if all values are basic types
        # at the below logic 

    if isinstance(data, bytes) and bytes_to_b64:
        return bytes_to_base64(data)

    elif isinstance(data, dict):
        tidied_data = {}
        _dumped_ids_copy = _dumped_ids.copy() if _dumped_ids is not None else set()
        data_id = id(data)
        if data_id not in _dumped_ids_copy:
            _dumped_ids_copy.add(data_id)
            for k, v in data.items():
                r = _internal_recursive_dump_to_basic_types(
                    data=v,
                    ignore_err=ignore_err,
                    include_non_dumpable=include_non_dumpable,
                    inner=True,
                    bytes_to_b64=bytes_to_b64,
                    no_recursion_limit=no_recursion_limit,
                    excepts=excepts,
                    _recursive_depth=_recursive_depth + 1,
                    _dumped_ids = _dumped_ids_copy,  # type: ignore
                )
                if r is not _SHOULD_NOT_INCLUDE:
                    tidied_data[k] = r

    elif isinstance(data, set) or (not isinstance(data, str) and isinstance(data, Sequence)):
        tidied_data = []
        _dumped_ids_copy = _dumped_ids.copy() if _dumped_ids is not None else set()
        data_id = id(data)
        if data_id not in _dumped_ids_copy:
            _dumped_ids_copy.add(data_id)
            for v in data:
                r = _internal_recursive_dump_to_basic_types(
                    data=v,
                    ignore_err=ignore_err,
                    include_non_dumpable=include_non_dumpable,
                    inner=True,
                    bytes_to_b64=bytes_to_b64,
                    no_recursion_limit=no_recursion_limit,
                    excepts=excepts,
                    _recursive_depth=_recursive_depth + 1,
                    _dumped_ids=_dumped_ids_copy,    # type: ignore
                )
                if r is not _SHOULD_NOT_INCLUDE:
                    tidied_data.append(r)
        if isinstance(data, set):
            tidied_data = set(tidied_data)
            
    elif hasattr(data, "__serialize__") and callable(data.__serialize__):  # type: ignore
        tidied_data = data.__serialize__()  # type: ignore
    else:
        tidied_data = data
    return tidied_data

def recursive_dump_to_basic_types(
    data: Any, 
    ignore_err=False, 
    include_non_dumpable=True, 
    bytes_to_b64=False,
    no_recursion_limit=False,
    excepts: type|Sequence[type]|None=None,
) -> BasicType:
    """
    recursively dump the data(BaseModel, attr class, ...) until reaching basic types,
    e.g. list, dict, str, int, float, etc.

    When `ignore_err`=False, non-dumpable data will:
        - (include_non_dumpable=True): be included in the result
        - (include_non_dumpable=False):
            - (when it is a sub-data of a dumpable data) ignore
            - (when it is the root data) still return the non-dumpable data
            
    This method will also prevent circular reference error in cases like:
    ```
    d1 = {'d2': {}}
    d1['d2']['d1'] = d1
    recursive_dump_to_basic_types(d1)   # -> {'d2': {}}, no error
    ```
    """
    _dumped_ids = set()
    if isinstance(excepts, type):
        excepts = (excepts,)
    return _internal_recursive_dump_to_basic_types(
        data, 
        ignore_err=ignore_err, 
        include_non_dumpable=include_non_dumpable, 
        bytes_to_b64=bytes_to_b64,
        no_recursion_limit=no_recursion_limit,
        excepts=(tuple(excepts) if excepts else None),
        _dumped_ids=_dumped_ids,    # type: ignore
    )  # type: ignore


def bytes_to_base64(data: bytes)->str:
    """change bytes to base64 string."""
    return base64.b64encode(data).decode("ascii")


def bytes_to_base64_chunks(data: bytes, chunk_size: int = 65535)->list[str]:
    """
    Change bytes to base64 string, and split it into chunks.
    This is usually be used for sending large data in SSE.
    """
    chunks: list[str] = []
    if len(data) > 65535:
        for i in range(0, len(data), chunk_size):
            chunks.append(bytes_to_base64(data[i : i + chunk_size]))
    else:
        chunks.append(bytes_to_base64(data))
    return chunks

def base64_to_bytes(data: str, encode='ascii')->bytes:
    """change base64 string to bytes."""
    return base64.b64decode(data.encode(encode))

def base64_str_chunks_to_bytes(datas: Iterable[str], encode="ascii")->bytes:
    """change base64 string to bytes."""
    all_bytes = [base64.b64decode(data.encode(encode)) for data in datas]
    all_bytes = b"".join(all_bytes)
    return all_bytes


__all__ = [
    "SerializableType",
    "is_serializable",
    "serialize",
    "deserialize",
    "recursive_dump_to_basic_types",
    "bytes_to_base64",
    "bytes_to_base64_chunks",
    'base64_to_bytes',
    "base64_str_chunks_to_bytes",
    "get_pydantic_type_adapter"
]


if __name__ == "__main__":
    def rec_dump_test():
        class A(BaseModel):
            a: 'A|None' = None
            
        a1 = A()
        a2 = A()
        a1.a = a2
        a2.a = a1
        
        print(recursive_dump_to_basic_types(a1))
        
        d = {'a': 1}
        lst = [d, d]
        print(recursive_dump_to_basic_types(lst))
        
    def adapter_test():
        a = get_pydantic_type_adapter(float)
        print(a.validate_python('1'))
    
    def deserialize_test():
        type X = Literal['a', 'b', 'c']     # type: ignore
        print(is_serializable(X))
        type Y = Annotated[X, 'test']       # type: ignore
        print(is_serializable(Y))
        
        def test(val, target_type=None):
            x = deserialize(val, target_type=target_type)   # type: ignore
            print(x, type(x))
        
        test('{"a": "1"}', target_type=dict)
        test('"1.0"', target_type=int)
        test('"1.0"', target_type=float)
        test('"1"',)
        
        test('"1"', target_type=bool)
        test('1', target_type=bool)
        test('0', target_type=bool)
        test('True', target_type=bool)
        test('true', target_type=bool)
        test('false', target_type=bool)
        test('123.321', target_type=int)

        class A(BaseModel):
            x: int = 1
        
        a = A()
        test(a.model_dump_json(), target_type=A|None)   # type: ignore
        test('null', target_type=A|None)
        
    # rec_dump_test()
    # adapter_test()
    deserialize_test()
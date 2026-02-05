'''override & modify from pymilvus.orm.iterator, to fixes its bugs & add async.'''

from pymilvus.orm.iterator import (QueryIterator as _QueryIterator, SearchIterator as _SearchIterator,
                                   iterator_cache, BATCH_SIZE, SearchPage, UNLIMITED, extend_batch_size,
                                   MAX_TRY_TIME)
from pymilvus.decorators import LOGGER
class QueryIterator(_QueryIterator):
    
    async def anext(self):
        cached_res = iterator_cache.fetch_cache(self._cache_id_in_use)
        ret = None
        if self.__is_res_sufficient(cached_res):    # type: ignore
            ret = cached_res[0 : self._kwargs[BATCH_SIZE]]  # type: ignore
            res_to_cache = cached_res[self._kwargs[BATCH_SIZE] :]   # type: ignore
            iterator_cache.cache(res_to_cache, self._cache_id_in_use)
        else:
            iterator_cache.release_cache(self._cache_id_in_use)
            current_expr = self.__setup_next_expr()
            res = await self._conn.query(   # type: ignore
                collection_name=self._collection_name,
                expr=current_expr,
                output_fields=self._output_fields,
                partition_names=self._partition_names,
                timeout=self._timeout,
                _async=True,
                **self._kwargs,
            )
            self.__maybe_cache(res)
            ret = res[0 : min(self._kwargs[BATCH_SIZE], len(res))]

        ret = self.__check_reached_limit(ret)
        self.__update_cursor(ret)
        self._returned_count += len(ret)
        return ret
    
    def __iter__(self):
        val = self.next()
        while val:
            yield val 
        self.close()
        
    async def __aiter__(self):
        val = await self.anext()
        while val:
            yield val
            val = await self.anext()
        self.close()


class SearchIterator(_SearchIterator):
    
    async def __async_execute_next_search(
        self, next_params: dict, next_expr: str, to_extend_batch: bool
    ) -> SearchPage:
        res = await self._conn.search(  # type: ignore
            self._iterator_params["collection_name"],
            self._iterator_params["data"],
            self._iterator_params["ann_field"],
            next_params,
            extend_batch_size(self._iterator_params[BATCH_SIZE], next_params, to_extend_batch),
            next_expr,
            self._iterator_params["partition_names"],
            self._iterator_params["output_fields"],
            self._iterator_params["round_decimal"],
            timeout=self._iterator_params["timeout"],
            schema=self._schema,
            _async=True, 
            **self._kwargs,
        )
        return SearchPage(res[0])
    
    async def __try_search_fill(self) -> SearchPage:
        final_page = SearchPage(None)   # type: ignore
        try_time = 0
        coefficient = 1
        while True:
            next_params = self.__next_params(coefficient)
            next_expr = self.__filtered_duplicated_result_expr(self._expr)  # type: ignore
            new_page = await self.__async_execute_next_search(next_params, next_expr, True)
            self.__update_filtered_ids(new_page)
            try_time += 1
            if len(new_page) > 0:
                final_page.merge(new_page.get_res())
                self._tail_band = new_page[-1].distance # type: ignore
            if len(final_page) >= self._iterator_params[BATCH_SIZE]:
                break
            if try_time > MAX_TRY_TIME:
                LOGGER.warning(f"Search probe exceed max try times:{MAX_TRY_TIME} directly break")
                break
            # if there's a ring containing no vectors matched, then we need to extend
            # the ring continually to avoid empty ring problem
            coefficient += 1
        return final_page
    
    async def anext(self):
        # 0. check reached limit
        if not self._init_success or self.__check_reached_limit():
            return SearchPage(None) # type: ignore
        ret_len = self._iterator_params[BATCH_SIZE]
        if self._limit is not UNLIMITED:
            left_len = self._limit - self._returned_count   # type: ignore
            ret_len = min(left_len, ret_len)

        # 1. if cached page is sufficient, directly return
        if self.__is_cache_enough(ret_len):
            ret_page = self.__extract_page_from_cache(ret_len)
            self._returned_count += len(ret_page)
            return ret_page

        # 2. if cached page not enough, try to fill the result by probing with constant width
        # until finish filling or exceeding max trial time: 10
        new_page = await self.__try_search_fill()
        cached_page_len = self.__push_new_page_to_cache(new_page)
        ret_len = min(cached_page_len, ret_len)
        ret_page = self.__extract_page_from_cache(ret_len)
        if len(ret_page) == self._iterator_params[BATCH_SIZE]:
            self.__update_width(ret_page)

        # 3. update filter ids to avoid returning result repeatedly
        self._returned_count += ret_len
        return ret_page
    
    def __iter__(self):
        val = self.next()
        while val.ids():
            yield val
        self.close()
        
    async def __aiter__(self):
        val = await self.anext()
        while val.ids():
            yield val
            val = await self.anext()
        self.close()


__all__ = ['QueryIterator', 'SearchIterator']
        
            

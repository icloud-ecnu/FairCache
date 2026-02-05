from collections import OrderedDict
from typing import Union, Dict, List
from time import time
from lmcache.logging import init_logger
from lmcache.storage_backend.evictor.base_evictor import BaseEvictor, PutStatus
from lmcache.utils import CacheEngineKey

from lmcache.storage_backend.mem_pool import CacheEntry

logger = init_logger(__name__)


class LRUClientEvictor(BaseEvictor):
    """
    LRU client-based cache evictor
    """

    def __init__(self, max_cache_size: float = 10.0, size_per_chunk: int = 1024**2):
        # The storage size limit (in bytes)
        self.MAX_CACHE_SIZE = int(max_cache_size * 1024**3)
        self.MAX_CACHE_SIZE_PER_CLIENT = int(max_cache_size * 1024**3 / 5)
        self.size_per_chunk = size_per_chunk

        # TODO(Jiayi): need a way to avoid fragmentation
        # current storage size (in bytes)
        self.current_cache_size = 0.0
        self.time_window = 100

    def update_on_get(self, key: Union[CacheEngineKey, str], client_id: int,
                      cache_dict: Dict[int, OrderedDict[CacheEngineKey, CacheEntry]]) -> None:
        """
        Update cache recency when a cache is used

        Input:
            key: a CacheEngineKey or a str
            cache_dict: a dict consists of current cache
        """
        cache_dict[client_id][key].update_access_time()
        cache_dict[client_id].move_to_end(key)

    # FIXME(Jiayi): comment out return type to bypass type checks
    # Need to align CacheEngineKey & str
    def update_on_put(self, cache_dict: Dict[int, OrderedDict[CacheEngineKey, CacheEntry]], cache_size: int, retrieved_keys: List[CacheEngineKey], client_id: int):
        """
        Evict cache when a new cache comes and the storage is full

        Input:
            cache_dict: a dict consists of current cache
            kv_obj: the new kv cache to be injected
        
        Return:
            evict_keys: a list of keys to be evicted
        """
        evict_keys = []
        
    
        if cache_size > self.MAX_CACHE_SIZE:
            logger.warning("Put failed due to limited cache storage")
            return None, [], PutStatus.ILLEGAL
        
        evict_client_id = None
        # 找出淘汰哪个client的数据
        # 只有当缓存空间不足时，才触发淘汰
        if cache_size + self.current_cache_size > self.MAX_CACHE_SIZE:
            evict_client_id = self.lru_client_to_evict(cache_dict, cache_size, client_id)
            # evict_client_id = self.mmf_client_to_evict(cache_dict, cache_size)
            iter_cache_dict = iter(cache_dict[evict_client_id])

        # evict cache until there's enough space
        while cache_size + self.current_cache_size > \
            self.MAX_CACHE_SIZE:
            evict_key = next(iter_cache_dict)
            if evict_key in retrieved_keys:
                return None, [], PutStatus.ILLEGAL
            evict_cache_size = self.get_size(cache_dict[evict_client_id][evict_key].kv_obj)
            self.current_cache_size -= evict_cache_size
            evict_keys.append(evict_key)

        # hack:  isolation policy
        # current_client_cache_size = 0
        # if client_id in cache_dict:
        #     for key, cache_entry in cache_dict[client_id].items():
        #         current_client_cache_size += self.get_size(cache_entry.kv_obj)
        # if current_client_cache_size + cache_size > self.MAX_CACHE_SIZE_PER_CLIENT:
        #     evict_client_id = client_id
        #     iter_cache_dict = iter(cache_dict[evict_client_id])

        # # evict cache until there's enough space
        # while cache_size + current_client_cache_size > \
        #     self.MAX_CACHE_SIZE_PER_CLIENT:
        #     evict_key = next(iter_cache_dict)
        #     if evict_key in retrieved_keys:
        #         return None, [], PutStatus.ILLEGAL
        #     evict_cache_size = self.get_size(cache_dict[evict_client_id][evict_key].kv_obj)
        #     self.current_cache_size -= evict_cache_size
        #     current_client_cache_size -= evict_cache_size
        #     evict_keys.append(evict_key)

        # update cache size
        self.current_cache_size += cache_size
        if len(evict_keys) > 0:
            logger.debug(
                f"Client {client_id} evicted {len(evict_keys)} chunks from client {evict_client_id}, "
                f"Current cache size: {self.current_cache_size} bytes, "
                f"Max cache size: {self.MAX_CACHE_SIZE} bytes")
        return evict_client_id, evict_keys, PutStatus.LEGAL
    

    def wac_client_to_evict(self, cache_dict: Dict[int, OrderedDict[CacheEngineKey, CacheEntry]], cache_size: int, current_client_id: int):
        """
        Find the client to evict
        wac policy: 
        """
        # 对各个client的冷数据大小和总数据大小进行记录
        evict_client_id = None
        cold_size_dict = {client_id: 0 for client_id in cache_dict.keys()}
        total_size_dict = {client_id: 0 for client_id in cache_dict.keys()}
        for client_id, cache in cache_dict.items():
            for key, cache_entry in cache.items():
                if time() - cache_entry.last_access_time > self.time_window:
                    cold_size_dict[client_id] += self.get_size(cache_entry.kv_obj)
                total_size_dict[client_id] += self.get_size(cache_entry.kv_obj)
        
        # 对各个client进行排序，首先按冷数据大小排序，其次按总数据大小排序
        sorted_clients = sorted(
            cold_size_dict.keys(), 
            key=lambda client_id: (cold_size_dict[client_id], total_size_dict[client_id]), 
            reverse=True
        )
        
        for client_id in sorted_clients:
            if total_size_dict[client_id] >= cache_size:
                evict_client_id = client_id
                break

        if evict_client_id is None:
            # 报错
            raise ValueError("No client to evict")
        
        if evict_client_id == current_client_id or current_client_id not in cache_dict:
            return evict_client_id
        else:
            if cold_size_dict[evict_client_id] > cold_size_dict[current_client_id]:
                return evict_client_id
            else:
                if total_size_dict[evict_client_id] - total_size_dict[current_client_id] <= self.size_per_chunk:
                    return current_client_id
                else:
                    return evict_client_id
    
    
    def mmf_client_to_evict(self, cache_dict: Dict[int, OrderedDict[CacheEngineKey, CacheEntry]], cache_size: int):
        """
        Find the client to evict
        mmf policy: 
        """
        # 对各个client的数据大小进行记录
        size_dict = {client_id: 0 for client_id in cache_dict.keys()}
        for client_id, cache in cache_dict.items():
            for key, cache_entry in cache.items():
                size_dict[client_id] += self.get_size(cache_entry.kv_obj)
        
        # 对各个client进行排序，从大到小排序
        sorted_clients = sorted(size_dict.keys(), key=lambda client_id: size_dict[client_id], reverse=True)
        for client_id in sorted_clients:
            if size_dict[client_id] >= cache_size:
                return client_id
        # 报错
        raise ValueError("No client to evict")
    
    def lru_client_to_evict(self, cache_dict: Dict[int, OrderedDict[CacheEngineKey, CacheEntry]], cache_size: int, current_client_id: int):
        """
        Find the client to evict
        lru policy: 
        """
        # 获取所有client的队首cacheentry的时间戳
        timestamps = {client_id: cache_entry.last_access_time for client_id, cache in cache_dict.items() if cache and (cache_entry := next(iter(cache.values())))}
        
        # 按照时间戳由旧到新排序client
        sorted_clients = sorted(timestamps.keys(), key=lambda client_id: timestamps[client_id])

        # 对各个client的数据大小进行记录
        size_dict = {client_id: 0 for client_id in cache_dict.keys()}
        for client_id, cache in cache_dict.items():
            for key, cache_entry in cache.items():
                size_dict[client_id] += self.get_size(cache_entry.kv_obj)
        
        for client_id in sorted_clients:
            if size_dict[client_id] >= cache_size:
                return client_id
        # 报错
        raise ValueError("No client to evict")
        
   


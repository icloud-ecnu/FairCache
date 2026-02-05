import heapq
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from time import time
import math

from lmcache.logging import init_logger
from lmcache.storage_backend.evictor.base_evictor import BaseEvictor, PutStatus
from lmcache.utils import CacheEngineKey
from lmcache.storage_backend.mem_pool import CacheEntry

logger = init_logger(__name__)


class FairEvictor(BaseEvictor):
    """
    Fair cache evictor based on value calculation and client cache usage.
    Uses P(t) = e^{-lambda * t} to calculate cache value.
    """

    def __init__(self, max_cache_size: float = 10.0, size_per_chunk: int = 1024**2, value_threshold: float = 0.01):
        # The storage size limit (in bytes)
        self.MAX_CACHE_SIZE = int(max_cache_size * 1024**3)
        self.size_per_chunk = size_per_chunk
        self.value_threshold = value_threshold  # Threshold for direct eviction
        
        # Current storage size (in bytes)
        self.current_cache_size = 0.0
        
        # Global cache dict: key -> CacheEntry
        # This will be passed from backend
        #
        # NOTE: 由于 lambda_val 只取有限个离散值（例如 API / Chat 两类），
        # 我们按 lambda_val 进行分组，为每一类维护一个最小堆。
        #
        # Per-client priority queues:
        #   client_heaps[client_id][lambda_val] -> [(value, key), ...]
        # 这里的 value 是“该 client 对该块的贡献价值”，用在按 client 淘汰时使用。
        self.client_heaps: Dict[int, Dict[float, List[Tuple[float, CacheEngineKey]]]] = {}
        
        # Global min heaps by lambda_val:
        #   global_heaps[lambda_val] -> [(value, key), ...]
        # 这里的 value 是该块的全局价值（所有 client 贡献的和），
        # 用在全局最小 value 淘汰时使用。
        self.global_heaps: Dict[float, List[Tuple[float, CacheEngineKey]]] = {}
        
        # Client cache usage: client_id -> cache_size (in bytes)
        # 较大的 size 代表更高的被淘汰优先级
        self.client_cache_sizes: Dict[int, float] = {}
        # 若之后需要真正的 max-heap，可以基于该 dict 构造，这里先不用额外结构
        self.client_size_heap: List[Tuple[float, int]] = []  # 预留，当前未使用
        
        # Mapping from key to its entry in heaps (for efficient updates)
        self.key_to_clients: Dict[CacheEngineKey, set] = {}  # key -> set of client_ids

    def _update_client_heap(self, client_id: int, lambda_val: float,
                            key: CacheEngineKey, value: float) -> None:
        """在指定 client / lambda_val 的最小堆中插入或更新一个条目（使用 lazy deletion）"""
        if client_id not in self.client_heaps:
            self.client_heaps[client_id] = {}
        if lambda_val not in self.client_heaps[client_id]:
            self.client_heaps[client_id][lambda_val] = []
        heap = self.client_heaps[client_id][lambda_val]
        # lazy deletion：不显式删除旧条目，依赖弹出时与 entry.value 对比过滤
        heapq.heappush(heap, (value, key))

    def _update_global_heap(self, lambda_val: float,
                             key: CacheEngineKey, value: float) -> None:
        """在指定 lambda_val 的全局最小堆中插入或更新一个条目（使用 lazy deletion）"""
        if lambda_val not in self.global_heaps:
            self.global_heaps[lambda_val] = []
        heap = self.global_heaps[lambda_val]
        heapq.heappush(heap, (value, key))

    def _update_client_cache_size(self, client_id: int, cache_size: float):
        """Update client's cache size"""
        self.client_cache_sizes[client_id] = cache_size
        # Update max heap (using negative for max heap)
        # We'll rebuild it when needed for simplicity

    def _get_max_client_id(self) -> Optional[int]:
        """Get the client with maximum cache usage"""
        if not self.client_cache_sizes:
            return None
        
        max_client_id = max(self.client_cache_sizes.keys(), 
                          key=lambda cid: self.client_cache_sizes[cid])
        return max_client_id

    def _get_min_value_key(
        self,
        cache_dict: OrderedDict[CacheEngineKey, CacheEntry],
        excluded: set[CacheEngineKey],
    ) -> Optional[CacheEngineKey]:
        """
        获取全局 value 最小的 cache key。
        利用按 lambda_val 分组的最小堆，并通过 lazy deletion 保证堆顶合法。
        """
        best_key: Optional[CacheEngineKey] = None
        best_value: float = float("inf")

        # 遍历每个 lambda 分组的堆，取各自合法堆顶的最小值
        for lambda_val, heap in self.global_heaps.items():
            while heap:
                value, key = heap[0]
                entry = cache_dict.get(key)
                # 条件 1：缓存已经被删
                # 条件 2：该 entry 的 lambda 已变
                # 条件 3：entry.value 已更新，当前堆顶是陈旧值
                if (
                    entry is None
                    or entry.lambda_val != lambda_val
                    or entry.value != value
                    or key in excluded
                ):
                    heapq.heappop(heap)
                    continue

                # 现在堆顶是合法的
                if value < best_value:
                    best_value = value
                    best_key = key
                break

        return best_key

    def _get_min_value_key_for_client(
        self,
        client_id: int,
        cache_dict: OrderedDict[CacheEngineKey, CacheEntry],
        excluded: set[CacheEngineKey],
    ) -> Optional[CacheEngineKey]:
        """
        获取某个 client 的“贡献 value”最小的 cache key。
        同样利用按 lambda_val 分组的最小堆，并通过 lazy deletion 保证堆顶合法。
        """
        if client_id not in self.client_heaps:
            return None

        best_key: Optional[CacheEngineKey] = None
        best_value: float = float("inf")

        for lambda_val, heap in self.client_heaps[client_id].items():
            while heap:
                value, key = heap[0]
                entry = cache_dict.get(key)
                if (
                    entry is None
                    or entry.lambda_val != lambda_val
                    or client_id not in entry.access_map
                    or key in excluded
                ):
                    # 该 client 已不再访问此块，或块被删 / lambda 变了，弹出
                    heapq.heappop(heap)
                    continue

                if value < best_value:
                    best_value = value
                    best_key = key
                break

        return best_key

    def update_on_get(self, key: CacheEngineKey, client_id: int,
                     cache_dict: OrderedDict[CacheEngineKey, CacheEntry]) -> None:
        """
        Update cache state when a cache is accessed.
        
        Input:
            key: CacheEngineKey
            client_id: ID of the client accessing the cache
            cache_dict: Global cache dictionary
        """
        if key not in cache_dict:
            return
        
        entry = cache_dict[key]
        
        # Check if this is a new client accessing an existing shared block
        was_new_client = client_id not in entry.access_map
        old_num_clients = len(entry.access_map)
        
        # Update access time for this client
        entry.update_access_time(client_id)
        
        # Update score (time-independent)
        entry.update_value()
        
        # Update key_to_clients mapping
        if key not in self.key_to_clients:
            self.key_to_clients[key] = set()
        self.key_to_clients[key].add(client_id)
        
        # Update global heap（使用全局 value）
        self._update_global_heap(entry.lambda_val, key, entry.value)

        # Update client heaps：所有已经访问过该块的 client 都共享同一个全局 value
        for cid in entry.access_map.keys():
            self._update_client_heap(cid, entry.lambda_val, key, entry.value)
        
        # Update client cache size (shared block: divide size by number of clients)
        cache_size = self.get_size(entry.kv_obj)
        num_clients = len(entry.access_map)
        
        if was_new_client and old_num_clients > 0:
            # This is a new client accessing an existing shared block
            # Need to update all clients' sizes
            old_per_client_size = cache_size / old_num_clients
            new_per_client_size = cache_size / num_clients
            
            # Update all existing clients (reduce their size)
            for other_client_id in list(entry.access_map.keys()):
                if other_client_id != client_id:
                    if other_client_id in self.client_cache_sizes:
                        self.client_cache_sizes[other_client_id] = max(0,
                            self.client_cache_sizes[other_client_id] - old_per_client_size + new_per_client_size)
                    else:
                        self.client_cache_sizes[other_client_id] = new_per_client_size
            
            # Add new client
            if client_id not in self.client_cache_sizes:
                self.client_cache_sizes[client_id] = 0.0
            self.client_cache_sizes[client_id] += new_per_client_size
        elif num_clients == 1:
            # This is the first client accessing this cache
            if client_id not in self.client_cache_sizes:
                self.client_cache_sizes[client_id] = 0.0
            self.client_cache_sizes[client_id] += cache_size
        # else: client already had access, just updating time, no size change

    def update_on_put(self, 
                     cache_dict: OrderedDict[CacheEngineKey, CacheEntry],
                     cache_size: int,
                     client_id: int,
                     lambda_val: float = 1.0,
                     retrieved_keys: Optional[List[CacheEngineKey]] = None) -> Tuple[Optional[int], List[CacheEngineKey], PutStatus]:
        """
        Evict cache when a new cache comes and the storage is full.
        
        Input:
            cache_dict: Global cache dictionary
            cache_size: Size of the new cache to be added
            client_id: ID of the client adding the cache
            lambda_val: Lambda value for value calculation (based on workload type)
            retrieved_keys: List of keys that were just retrieved (should not be evicted)
        
        Return:
            (evict_client_id, evict_keys, PutStatus)
        """
        if retrieved_keys is None:
            retrieved_keys = []
        
        evict_keys: List[CacheEngineKey] = []
        evict_client_id = None
        
        # Check if cache is too large
        if cache_size > self.MAX_CACHE_SIZE:
            logger.warning("Put failed due to limited cache storage")
            return None, [], PutStatus.ILLEGAL
        
        current_time = time()
        # 记录本轮已经选择淘汰的 key，避免在同一轮中重复选中
        excluded: set[CacheEngineKey] = set()
        
        # Step 1: Evict caches with value below threshold
        while cache_size + self.current_cache_size > self.MAX_CACHE_SIZE:
            min_key = self._get_min_value_key(cache_dict, excluded)
            if min_key is None or min_key in retrieved_keys:
                break
            min_entry = cache_dict[min_key]
            # 真实的 P(t)（与 score 单调等价，但用于阈值判断）
            real_value = min_entry.instant_value(current_time)
            if real_value >= self.value_threshold:
                # No more caches below threshold
                break
            
            # Direct eviction
            evict_keys.append(min_key)
            evict_cache_size = self.get_size(min_entry.kv_obj)
            self.current_cache_size -= evict_cache_size
            excluded.add(min_key)
            
            # Update client cache sizes
            num_clients = len(min_entry.access_map)
            if num_clients > 0:
                per_client_size = evict_cache_size / num_clients
                for cid in min_entry.access_map:
                    if cid in self.client_cache_sizes:
                        self.client_cache_sizes[cid] = max(0,
                            self.client_cache_sizes[cid] - per_client_size)
            
            # Remove from key_to_clients
            if min_key in self.key_to_clients:
                del self.key_to_clients[min_key]
            
            # 实际从 cache_dict 删除由 backend 的 remove 完成；这里仅维护 evictor 内部状态
        
        # Step 2: If still need more space, evict from max client
        while cache_size + self.current_cache_size > self.MAX_CACHE_SIZE:
            max_client_id = self._get_max_client_id()
            if max_client_id is None:
                logger.warning("No client to evict from")
                return None, [], PutStatus.ILLEGAL
            
            # Find minimum value cache for this client
            min_key_for_client = self._get_min_value_key_for_client(max_client_id, cache_dict, excluded)
            
            if min_key_for_client is None or min_key_for_client in retrieved_keys:
                # Cannot evict from this client, try next
                # Remove this client temporarily and try again
                if max_client_id in self.client_cache_sizes:
                    del self.client_cache_sizes[max_client_id]
                continue
            
            evict_entry = cache_dict[min_key_for_client]
            evict_cache_size = self.get_size(evict_entry.kv_obj)
            
            evict_keys.append(min_key_for_client)
            self.current_cache_size -= evict_cache_size
            evict_client_id = max_client_id
            excluded.add(min_key_for_client)
            
            # Update client cache sizes
            num_clients = len(evict_entry.access_map)
            if num_clients > 0:
                per_client_size = evict_cache_size / num_clients
                for cid in evict_entry.access_map:
                    if cid in self.client_cache_sizes:
                        self.client_cache_sizes[cid] = max(0,
                            self.client_cache_sizes[cid] - per_client_size)
            
            # Remove from key_to_clients
            if min_key_for_client in self.key_to_clients:
                del self.key_to_clients[min_key_for_client]
        
        # Update cache size
        self.current_cache_size += cache_size
        
        if len(evict_keys) > 0:
            logger.debug(
                f"Client {client_id} evicted {len(evict_keys)} chunks, "
                f"Current cache size: {self.current_cache_size} bytes, "
                f"Max cache size: {self.MAX_CACHE_SIZE} bytes")
        
        return evict_client_id, evict_keys, PutStatus.LEGAL
    
    def register_new_cache(self, key: CacheEngineKey, entry: CacheEntry, 
                           client_id: int, cache_dict: OrderedDict[CacheEngineKey, CacheEntry]):
        """
        Register a new cache entry in the evictor's data structures.
        Called after a new cache is added.
        
        Note: current_cache_size should already be updated by update_on_put,
        so we don't update it here.
        """
        current_time = time()
        # 先设置该 client 的访问时间，再计算 time-independent 的 score
        entry.update_access_time(client_id)
        entry.update_value()
        
        # Update key_to_clients
        if key not in self.key_to_clients:
            self.key_to_clients[key] = set()
        self.key_to_clients[key].add(client_id)
        
        # Update global heap（全局 score）
        self._update_global_heap(entry.lambda_val, key, entry.value)

        # Update client heap（当前 client 的 score 与全局相同）
        self._update_client_heap(client_id, entry.lambda_val, key, entry.value)
        
        # Update client cache size
        # For a new cache, the client gets full size initially
        cache_size = self.get_size(entry.kv_obj)
        if client_id not in self.client_cache_sizes:
            self.client_cache_sizes[client_id] = 0.0
        self.client_cache_sizes[client_id] += cache_size

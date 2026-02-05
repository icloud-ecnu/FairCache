import os
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Union
import json

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file

from lmcache.config import LMCacheEngineConfig, LMCacheMemPoolMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.storage_backend.evictor import LRUEvictor, LRUClientEvictor
from lmcache.storage_backend.evictor.fair_evictor import FairEvictor
from lmcache.storage_backend.evictor.base_evictor import PutStatus
from lmcache.storage_backend.mem_pool import (KVObj, CacheEntry, LocalCPUBufferPool,
                                              LocalCPUPool, LocalGPUPool,
                                              LocalPool)
from lmcache.utils import (CacheEngineKey, DiskCacheMetadata, KVCache,
                           _lmcache_nvtx_annotate)
from lmcache.storage_backend.mem_pool import CacheEntry
import contextvars

client_id_ctx = contextvars.ContextVar("client_id", default=None)
request_type_ctx = contextvars.ContextVar("request_type", default="api")

# Two lambda values for different request types
LAMBDA_HIGH = 0.1  # API 请求：更快衰减
LAMBDA_LOW = 0.01  # Chat 请求：更慢衰减

logger = init_logger(__name__)


class LocalBackendEndSignal:
    pass


class LMCLocalBackend(LMCBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local cpu/gpu
    memory.
    """

    def __init__(self,
                 config: LMCacheEngineConfig,
                 metadata: LMCacheMemPoolMetadata,
                 dst_device: str = "cuda"):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
                configuration
        """
        super().__init__(dst_device)

        self.chunk_size = config.chunk_size
        self.config = config
        self.dict: OrderedDict[CacheEngineKey, KVObj] = OrderedDict()
        self.device = config.local_device

        self.put_queue: queue.Queue[
            Union[Tuple[CacheEngineKey, torch.Tensor],
                  LocalBackendEndSignal]] = queue.Queue()
        self.put_thread = threading.Thread(target=self.put_worker, args=())
        self.put_thread.start()
        self.update_lock = threading.Lock()

        # TODO(Jiayi): The storage size and caching policy for both
        # evictor and mpool need to be configured dynamically
        max_cache_size = self.config.max_local_cache_size
        self.evictor = LRUEvictor(max_cache_size)
        self.mpool: LocalPool
        if self.device == "cpu":
            self.mpool = LocalCPUPool(metadata)
        elif self.device == "cuda":
            self.mpool = LocalGPUPool(metadata)

        # TODO(Jiayi): A gpu buffer could speed up `get`
        # self.fix_sized_dst_buffer = torch.tensor()

    def contains(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        return key in self.dict

    def remove(
        self,
        key: CacheEngineKey,
    ) -> None:
        """
        Remove the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format

        """
        kv_obj = self.dict.pop(key)
        self.mpool.free(kv_obj)

    @_lmcache_nvtx_annotate
    def put_worker(self, ):
        while True:
            item = self.put_queue.get()
            if isinstance(item, LocalBackendEndSignal):
                break
            key, value = item
            self.put_nonblocking(key, value)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def put_nonblocking(self, key, kv_chunk):
        # Obtain keys to evict
        self.update_lock.acquire()
        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, self.mpool.size_per_chunk)
        if put_status == PutStatus.ILLEGAL:
            self.update_lock.release()
            return

        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)

        # free old block to avoid mem leak
        if key in self.dict:
            self.remove(key)

        # Allocate the kv chunk
        kv_obj = self.mpool.allocate(kv_chunk)
        self.update_lock.release()

        if kv_obj is None:
            return

        put_stream = torch.cuda.Stream()
        if kv_chunk.device != torch.cpu:
            # wait operation in main stream to finish
            # e.g., view operations on kv_chunk
            put_stream.wait_stream(torch.cuda.default_stream(kv_chunk.device))

        with torch.cuda.stream(put_stream):
            kv_obj.data.copy_(kv_chunk, non_blocking=True)
            kv_chunk.record_stream(put_stream)
        put_stream.synchronize()

        # Store new chunk
        self.update_lock.acquire()
        self.dict[key] = kv_obj
        self.update_lock.release()

    @torch.inference_mode()
    def put_blocking(self, key, kv_chunk):

        # Obtain keys to evict
        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, self.mpool.size_per_chunk)

        # Abort put if cache too big
        if put_status == PutStatus.ILLEGAL:
            return

        # free old block to avoid mem leak
        if key in self.dict:
            self.remove(key)

        # Evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)

        kv_obj = self.mpool.allocate(kv_chunk)

        if kv_obj is None:
            return

        kv_obj.data.copy_(kv_chunk, non_blocking=False)

        # Store new chunk
        self.dict[key] = kv_obj

    def put(
        self,
        key: CacheEngineKey,
        kv_chunk: torch.Tensor,
        blocking: bool = True,
    ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested 
            tuples

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        if blocking:
            self.put_blocking(key, kv_chunk)
        else:
            self.put_queue.put((key, kv_chunk))

    @_lmcache_nvtx_annotate
    def get(
        self,
        key: CacheEngineKey,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output:
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        kv_chunk = None

        self.update_lock.acquire()
        kv_obj = self.dict.get(key, None)

        # Update cache recency
        if kv_obj is not None:
            self.evictor.update_on_get(key, self.dict)
            kv_chunk = kv_obj.data.to(self.dst_device)

        self.update_lock.release()

        return kv_chunk

    def close(self):
        if self.put_thread is not None and self.put_thread.is_alive():
            self.put_queue.put(LocalBackendEndSignal())
            self.put_thread.join()
            logger.info("Closed the put worker in local backend")

    def __del__(self):
        self.close()


# TODO(Jiayi): need to optimize disk saving/loading
# current impl. with "safetensors" might not be efficient
# but it is better than "torch.save/load"


@_lmcache_nvtx_annotate
@torch.inference_mode()
def save_disk(
    path: str,
    kv_chunk: torch.Tensor,
):
    save_file({"kv_chunk": kv_chunk.contiguous()}, path)


class LMCLocalDiskBackend(LMCBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local disk.
    """

    def __init__(self,
                 config: LMCacheEngineConfig,
                 metadata: LMCacheMemPoolMetadata,
                 dst_device: str = "cuda"):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
                configuration
        """
        super().__init__(dst_device)

        self.chunk_size = config.chunk_size
        self.config = config
        self.dict: OrderedDict[CacheEngineKey,
                               DiskCacheMetadata] = OrderedDict()
        self.path = config.local_device

        assert self.path is not None, ("Need to specify local path if when "
                                       "using LMCLocalDiskBackend")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.update_lock = threading.Lock()

        self.put_queue: queue.Queue[
            Union[Tuple[CacheEngineKey, torch.Tensor],
                  LocalBackendEndSignal]] = queue.Queue()
        self.put_thread = threading.Thread(target=self.put_worker, args=())
        self.put_thread.start()

        self.future_pool: Dict[CacheEngineKey, Tuple[Future, KVObj]] = {}
        self.stop_event = threading.Event()
        self.sweeper_thread = threading.Thread(target=self.buffer_sweeper,
                                               args=())
        self.sweeper_thread.start()

        # TODO(Jiayi): The storage size and caching policy for both
        # evictor and mpool need to be configured dynamically
        self.evictor = LRUEvictor(config.max_local_cache_size)
        # NOTE(Jiayi): This mbufferpool should be smaller than the actual
        # cpu backend but big enough to avoid stalls in save
        # TODO(Jiayi): share the buffer if both cpu and disk backend are enabled
        self.cpu_mbufferpool = LocalCPUBufferPool(metadata)

        self.proc_pool_executor = ProcessPoolExecutor(max_workers=4)

    def contains(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        with self.update_lock:
            return key in self.dict

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        """
        Convert key to path_name

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            returns the path name
        """
        return self.path + key.to_string().replace("/", "-") + ".pt"

    def remove(
        self,
        key: CacheEngineKey,
    ) -> None:
        """
        Remove the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format

        """

        path = self.dict[key].path
        self.dict.pop(key)
        os.remove(path)

    @_lmcache_nvtx_annotate
    def put_worker(self, ):
        while True:
            item = self.put_queue.get()
            if isinstance(item, LocalBackendEndSignal):
                break
            key, value = item
            self.put_nonblocking(key, value)

    def buffer_sweeper(self, ):
        """
        Sweep the future pool to free up memory.
        """
        while not self.stop_event:
            logger.debug("Sweeping memory buffer")
            self.update_lock.acquire()
            for key in list(self.future_pool.keys()):
                future = self.future_pool[key][0]
                kv_obj = self.future_pool[key][1]
                if not future.done():
                    continue
                self.cpu_mbufferpool.free(kv_obj)
                del self.future_pool[key]
            self.update_lock.release()

            # sweep the memory every 30s
            time.sleep(30)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def put_nonblocking(
        self,
        key: CacheEngineKey,
        kv_chunk: torch.Tensor,
    ) -> None:
        path = self._key_to_path(key)
        logger.debug(f"Saving cache to {path}")

        self.update_lock.acquire()

        # Skip store if task is already being executed
        # TODO(Jiayi): what if already stored, should we
        # overwrite or skip?
        if key in self.future_pool:
            self.update_lock.release()
            return

        # Obtain keys to evict
        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, self.evictor.get_size(kv_chunk))

        # Abort put if cache too big
        if put_status == PutStatus.ILLEGAL:
            self.update_lock.release()
            return

        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)
        self.update_lock.release()

        kv_obj = None

        # Allocate the kv chunk
        while kv_obj is None:
            self.update_lock.acquire()
            kv_obj = self.cpu_mbufferpool.allocate(kv_chunk)
            self.update_lock.release()
            if kv_obj is None:
                # TODO(Jiayi): Please tune the sleep time for better performance
                time.sleep(0.01)

        put_stream = torch.cuda.Stream()
        put_stream.wait_stream(torch.cuda.default_stream(kv_chunk.device))
        with torch.cuda.stream(put_stream):
            kv_obj.data.copy_(kv_chunk, non_blocking=True)
            kv_chunk.record_stream(put_stream)
        put_stream.synchronize()

        future = self.proc_pool_executor.submit(save_disk, path, kv_obj.data)

        self.update_lock.acquire()
        self.future_pool[key] = (future, kv_obj)
        self.dict[key] = DiskCacheMetadata(path, kv_obj.size)
        # NOTE(Jiayi): the following `free` will result in data corruption
        # The serialized object (`kv_obj.data` in `submit`) may reference
        # the external memory (cpu tensor might be shared in multiprocessing),
        # and if the tensor is deleted, it might be invalidated.
        # self.cpu_mbufferpool.free(kv_obj)
        self.update_lock.release()

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def put_blocking(
        self,
        key: CacheEngineKey,
        kv_chunk: torch.Tensor,
    ) -> None:
        path = self._key_to_path(key)
        logger.debug(f"Saving cache to {path}")

        self.update_lock.acquire()
        # Obtain keys to evict
        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, self.evictor.get_size(kv_chunk))

        # Abort put if cache too big
        if put_status == PutStatus.ILLEGAL:
            self.update_lock.release()
            return

        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)
        self.update_lock.release()

        # The following order matters of `save_file` and `update dictionary`
        # matters
        save_file({"kv_chunk": kv_chunk}, path)

        self.update_lock.acquire()
        self.dict[key] = DiskCacheMetadata(path,
                                           self.evictor.get_size(kv_chunk))
        self.update_lock.release()

    def put(
        self,
        key: CacheEngineKey,
        kv_chunk: torch.Tensor,
        blocking: bool = True,
    ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested 
            tuples

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        if blocking:
            self.put_blocking(key, kv_chunk)
        else:
            self.put_queue.put((key, kv_chunk))

    @_lmcache_nvtx_annotate
    def get(
        self,
        key: CacheEngineKey,
    ) -> Optional[KVCache]:
        """
        Retrieve the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output:
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        self.update_lock.acquire()
        if key not in self.dict:
            self.update_lock.release()
            return None

        if key in self.future_pool:
            future = self.future_pool[key][0]
            kv_obj = self.future_pool[key][1]

            # NOTE(Jiayi): the following code is blocking
            # if future.exception():
            #   raise Exception(f"Task raised an exception: \
            #   {future.exception()}")
            if not future.done():
                self.update_lock.release()
                return None
            self.cpu_mbufferpool.free(kv_obj)
            del self.future_pool[key]

        path = self.dict[key].path
        self.evictor.update_on_get(key, self.dict)

        with safe_open(path, framework="pt",
                       device=self.dst_device) as f:  # type: ignore
            kv_chunk = f.get_tensor("kv_chunk")
        self.update_lock.release()
        return kv_chunk

    def close(self):
        if self.put_thread is not None and self.put_thread.is_alive():
            self.put_queue.put(LocalBackendEndSignal())
            self.put_thread.join()

        if self.sweeper_thread is not None and self.sweeper_thread.is_alive():
            self.stop_event.set()
            self.sweeper_thread.join()

        self.proc_pool_executor.shutdown()
        logger.info("Closed the workers in local disk backend")

    def __del__(self):
        self.close()



class LMCLocalFairBackend(LMCBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local cpu/gpu
    memory with fair caching support.
    Uses global key-value storage to support shared blocks.
    """

    def __init__(self,
                 config: LMCacheEngineConfig,
                 metadata: LMCacheMemPoolMetadata,
                 dst_device: str = "cuda"):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
                configuration
        """
        super().__init__(dst_device)

        self.chunk_size = config.chunk_size
        self.config = config
        # Global key-value storage to support shared blocks
        self.dict: OrderedDict[CacheEngineKey, CacheEntry] = OrderedDict()
        self.device = config.local_device

        self.put_queue: queue.Queue[
            Union[Tuple[int, CacheEngineKey, torch.Tensor, Optional[List[CacheEngineKey]], float],
                  LocalBackendEndSignal]] = queue.Queue()
        self.put_thread = threading.Thread(target=self.put_worker, args=())
        self.put_thread.start()
        self.update_lock = threading.Lock()

        # TODO(Jiayi): The storage size and caching policy for both
        # evictor and mpool need to be configured dynamically
        max_cache_size = self.config.max_local_cache_size

        self.mpool: LocalPool
        if self.device == "cpu":
            self.mpool = LocalCPUPool(metadata)
        elif self.device == "cuda":
            self.mpool = LocalGPUPool(metadata)
        
        self.evictor = FairEvictor(max_cache_size, self.mpool.size_per_chunk)

        # TODO(Jiayi): A gpu buffer could speed up `get`
        # self.fix_sized_dst_buffer = torch.tensor()

    def contains(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        with self.update_lock:
            return key in self.dict

    def remove(
        self,
        key: CacheEngineKey,
    ) -> None:
        """
        Remove the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format

        Note:
            This method updates client_cache_sizes but NOT current_cache_size.
            current_cache_size is managed by evictor's update_on_put method.
        """
        if key not in self.dict:
            return
        cache_entry = self.dict.pop(key)
        self.mpool.free(cache_entry.kv_obj)
        
        # Update client cache sizes in evictor
        cache_size = self.evictor.get_size(cache_entry.kv_obj)
        num_clients = len(cache_entry.access_map)
        if num_clients > 0:
            per_client_size = cache_size / num_clients
            for client_id in cache_entry.access_map:
                if client_id in self.evictor.client_cache_sizes:
                    self.evictor.client_cache_sizes[client_id] = max(0,
                        self.evictor.client_cache_sizes[client_id] - per_client_size)
        
        # Remove from key_to_clients
        if key in self.evictor.key_to_clients:
            del self.evictor.key_to_clients[key]

    @_lmcache_nvtx_annotate
    def put_worker(self, ):
        while True:
            item = self.put_queue.get()
            if isinstance(item, LocalBackendEndSignal):
                break
            client_id, key, value, retrieved_keys, lambda_val = item
            self.put_nonblocking(client_id, key, value, retrieved_keys, lambda_val)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def put_nonblocking(self, client_id, key, kv_chunk, retrieved_keys, lambda_val):
        # free old block to avoid mem leak
        self.update_lock.acquire()
        old_cache_size = 0
        if key in self.dict:
            old_entry = self.dict[key]
            # Calculate old cache size before removing
            old_cache_size = self.evictor.get_size(old_entry.kv_obj)
            # Remove old entry (this updates client_cache_sizes but not current_cache_size)
            self.remove(key)
            # Update current_cache_size manually
            self.evictor.current_cache_size -= old_cache_size
        
        # Obtain keys to evict
        cache_size = self.evictor.get_size(kv_chunk)
        evict_client_id, evict_keys, put_status = self.evictor.update_on_put(
            self.dict, cache_size, client_id, lambda_val, retrieved_keys)
        if put_status == PutStatus.ILLEGAL:
            # Restore old cache size if put failed
            if old_cache_size > 0:
                self.evictor.current_cache_size += old_cache_size
            self.update_lock.release()
            return

        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)

        # Allocate the kv chunk
        kv_obj = self.mpool.allocate(kv_chunk)
        self.update_lock.release()

        if kv_obj is None:
            # Restore old cache size if allocation failed
            if old_cache_size > 0:
                self.update_lock.acquire()
                self.evictor.current_cache_size += old_cache_size
                self.update_lock.release()
            return

        put_stream = torch.cuda.Stream()
        if kv_chunk.device != torch.cpu:
            # wait operation in main stream to finish
            # e.g., view operations on kv_chunk
            put_stream.wait_stream(torch.cuda.default_stream(kv_chunk.device))

        with torch.cuda.stream(put_stream):
            kv_obj.data.copy_(kv_chunk, non_blocking=True)
            kv_chunk.record_stream(put_stream)
        put_stream.synchronize()

        # Store new chunk
        self.update_lock.acquire()
        cache_entry = CacheEntry(kv_obj, lambda_val=lambda_val)
        self.dict[key] = cache_entry
        # Register in evictor (this will update current_cache_size)
        self.evictor.register_new_cache(key, cache_entry, client_id, self.dict)
        self.update_lock.release()


    @torch.inference_mode()
    def put_blocking(self, client_id, key, kv_chunk, retrieved_keys, lambda_val):
        self.update_lock.acquire()
        
        # free old block to avoid mem leak
        old_cache_size = 0
        if key in self.dict:
            old_entry = self.dict[key]
            # Calculate old cache size before removing
            old_cache_size = self.evictor.get_size(old_entry.kv_obj)
            # Remove old entry (this updates client_cache_sizes but not current_cache_size)
            self.remove(key)
            # Update current_cache_size manually
            self.evictor.current_cache_size -= old_cache_size

        # Obtain keys to evict
        cache_size = self.evictor.get_size(kv_chunk)
        evict_client_id, evict_keys, put_status = self.evictor.update_on_put(
            self.dict, cache_size, client_id, lambda_val, retrieved_keys)

        # Abort put if cache too big
        if put_status == PutStatus.ILLEGAL:
            # Restore old cache size if put failed
            if old_cache_size > 0:
                self.evictor.current_cache_size += old_cache_size
            self.update_lock.release()
            return

        # Evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)

        kv_obj = self.mpool.allocate(kv_chunk)
        self.update_lock.release()

        if kv_obj is None:
            # Restore old cache size if allocation failed
            if old_cache_size > 0:
                self.update_lock.acquire()
                self.evictor.current_cache_size += old_cache_size
                self.update_lock.release()
            return

        kv_obj.data.copy_(kv_chunk, non_blocking=False)

        # Store new chunk
        self.update_lock.acquire()
        cache_entry = CacheEntry(kv_obj, lambda_val=lambda_val)
        self.dict[key] = cache_entry
        # Register in evictor (this will update current_cache_size)
        self.evictor.register_new_cache(key, cache_entry, client_id, self.dict)
        self.update_lock.release()

    def put(
        self,
        key: CacheEngineKey,
        kv_chunk: torch.Tensor,
        blocking: bool = True,
        retrieved_keys: Optional[List[CacheEngineKey]] = [],
        lambda_val: Optional[float] = None,
    ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested 
            tuples
            blocking: whether to block until put completes
            retrieved_keys: list of keys that were just retrieved (should not be evicted)
            lambda_val: lambda value for value calculation (based on workload type)

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        client_id = client_id_ctx.get()
        if client_id is None or client_id == -1:
            return
        # 根据请求类型决定 lambda，如果调用方未显式传入
        if lambda_val is None:
            req_type = request_type_ctx.get()
            if req_type == "chat":
                lambda_val = LAMBDA_LOW
            else:
                lambda_val = LAMBDA_HIGH
        if blocking:
            self.put_blocking(client_id, key, kv_chunk, retrieved_keys, lambda_val)
        else:
            self.put_queue.put((client_id, key, kv_chunk, retrieved_keys, lambda_val))
        
        # 仅让一个 GPU（或者 rank=0 那个 GPU）执行输出
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.output_cache_size_of_each_client()

    @_lmcache_nvtx_annotate
    def get(
        self,
        key: CacheEngineKey,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve the KV cache chunk by the given key.

        注意：对于 Fair 后端，我们不在这里更新 evictor / access_map，
        而是统一由 batch 级别的 update_after_get 来更新。
        这样可以保证 get_attribution_info 看到的是「更新前」的 access_map，
        用于判断当前 client 是否为新加入的共享者。
        """
        kv_chunk = None
        client_id = client_id_ctx.get()
        
        if client_id is None or client_id == -1:
            return None

        self.update_lock.acquire()
        cache_entry = self.dict.get(key, None)

        # 仅读取数据，不更新 access_map；由 update_after_get 统一更新
        if cache_entry is not None:
            kv_chunk = cache_entry.kv_obj.data.to(self.dst_device)

        self.update_lock.release()

        return kv_chunk

    def close(self):
        if self.put_thread is not None and self.put_thread.is_alive():
            self.put_queue.put(LocalBackendEndSignal())
            self.put_thread.join()
            logger.info("Closed the put worker in local backend")

    def __del__(self):
        self.close()


    def get_attribution_info(
        self, keys: List[CacheEngineKey]
    ) -> List[Tuple[bool, List[int]]]:
        """
        For each key, return (was_new_client, old_client_ids) *before* any
        update. Used for dynamic cost attribution: when current client hits
        a block they did not create, we adjust hit/miss and issue credits to
        old clients. Must be called with client_id in context, and *before*
        update_after_get.
        """
        client_id = client_id_ctx.get()
        if client_id is None or client_id == -1:
            return [(False, []) for _ in keys]
        result = []
        self.update_lock.acquire()
        try:
            for key in keys:
                entry = self.dict.get(key)
                if entry is None:
                    result.append((False, []))
                else:
                    was_new_client = client_id not in entry.access_map
                    old_client_ids = list(entry.access_map.keys())
                    result.append((was_new_client, old_client_ids))
        finally:
            self.update_lock.release()
        return result

    def update_after_get(self, retrieved_keys):
        """Update evictor state after retrieving multiple keys"""
        client_id = client_id_ctx.get()
        if client_id is None or client_id == -1:
            return
        self.update_lock.acquire()
        for key in reversed(retrieved_keys):
            if key in self.dict:
                self.evictor.update_on_get(key, client_id, self.dict)
        self.update_lock.release()

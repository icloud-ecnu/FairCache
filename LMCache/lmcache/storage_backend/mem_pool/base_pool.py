import abc
from dataclasses import dataclass, field
from typing import Optional, Dict
from time import time
import math

import torch


@dataclass
class KVObj:
    chunk_idx: int
    size: int  # size of the obj in bytes
    data: torch.Tensor

@dataclass
class CacheEntry:
    kv_obj: KVObj
    last_access_time: float = field(default_factory=time)
    # For fair caching: access_map stores last access time for each client
    access_map: Dict[int, float] = field(default_factory=dict)
    # Lambda value for value calculation (based on workload type)
    lambda_val: float = 1.0
    # Current score of this cache entry (time-independent ordering metric)
    # 对于给定 lambda_val，这个 score 与任意时刻的 P(t) 单调等价，
    # 但不依赖 current time，从而避免频繁全表更新。
    value: float = 0.0
    
    def update_access_time(self, client_id: Optional[int] = None):
        """Update access time for a specific client or all clients"""
        current_time = time()
        self.last_access_time = current_time
        if client_id is not None:
            self.access_map[client_id] = current_time
    
    def calculate_value(self) -> float:
        """
        计算用于排序的 score（time-independent）。
        对于给定 lambda_val，与真实的 P(t) = sum e^{-lambda * (t - t_i)} 单调等价：
        令 S = sum e^{lambda * t_i}，则 P(t) = e^{-lambda * t} * S。
        这里返回 log S，以避免指数溢出，且保持与 S 相同的排序。
        """
        if not self.access_map:
            # 没有 client 访问过时，用 last_access_time 近似一个“自访问”
            a = self.lambda_val * self.last_access_time
            return a  # log(exp(a)) = a
        
        # 稳定的 log-sum-exp: log(sum_i exp(a_i)), a_i = lambda * t_i
        a_vals = [self.lambda_val * last_access
                  for last_access in self.access_map.values()]
        m = max(a_vals)
        # sum exp(a_i - m) 不会溢出，因为 (a_i - m) <= 0
        sum_exp = 0.0
        for a in a_vals:
            sum_exp += math.exp(a - m)
        return m + math.log(sum_exp)
    
    def update_value(self):
        """更新 time-independent 的 score，用于堆排序"""
        self.value = self.calculate_value()

    def instant_value(self, current_time: float) -> float:
        """
        计算真实的 P(t) = sum e^{-lambda * (t - t_i)}，用于阈值判断等。
        """
        if not self.access_map:
            t = current_time - self.last_access_time
            return math.exp(-self.lambda_val * t)
        
        total = 0.0
        for _, last_access in self.access_map.items():
            t = current_time - last_access
            total += math.exp(-self.lambda_val * t)
        return total


class BasePool(metaclass=abc.ABCMeta):
    """
    Interface for mem pool
    """

    @abc.abstractmethod
    def allocate(self, kv_chunk: torch.Tensor) -> Optional[KVObj]:
        """
        Allocate a buffer memory pointer from the memory pool.
        
        Input:
            kv_chunk: the kv tensor to be stored
        
        Returns:
            KVObj with a memory pointer (torch tensor view).
            None if memory is full.
        
        Note:
            This does not perform the actual memory movement.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def free(self, kv_obj: KVObj):
        """
        Free the corresponding memory chunk
        
        Input:
            the KVObj to be freed
        """
        raise NotImplementedError

from collections import deque, Counter
from typing import List, Dict, TypeAlias, Deque, Optional
from vllm.sequence import SequenceGroup
import json
import time

client_id_type: TypeAlias = int


# Cumulative Prefill cost and credits for fair scheduling (CPI-driven)
class CPIReqQueue():

    def __init__(self, hit_weight=0.02, prefill_weight=0.1, decode_weight=0.2, eta: float = 1.0) -> None:
        # weight
        self.hit_weight = hit_weight
        self.prefill_weight = prefill_weight
        self.decode_weight =  decode_weight
        self.eta = eta  # attribution strength for dynamic cost attribution

        # 各用户CPI使用量记分板
        self.served: Dict[client_id_type, float] = {}

        # 各用户累积信用（用于 prefix 创建者补偿）
        self.credits: Dict[client_id_type, float] = {}

        # 各用户的队列
        self.client_waiting_queues: Dict[client_id_type, Deque[SequenceGroup]] = {}

        # 各用户各调度了多少请求
        self.served_requests: Dict[client_id_type, int] = {}

        # 各用户各调度了多少token
        self.served_tokens: Dict[client_id_type, int] = {}

        # 各用户命中token数量
        self.hit_tokens_count: Dict[client_id_type, int] = {}

        # 各用户miss token数量
        self.miss_tokens_count:  Dict[client_id_type, int] = {}

    # 新到来的请求进入waiting队列，加在队列尾部
    def append(self, seq_group: SequenceGroup):
        # 请求入队
        # client刚出现，新建一个该client对应的deque
        if seq_group.client_id not in self.client_waiting_queues:
            self.client_waiting_queues[seq_group.client_id] = deque([seq_group])
            self.served[seq_group.client_id] = 0
        # client已经存在
        else:
            self.client_waiting_queues[seq_group.client_id].append(seq_group)

        # waiting queue of the client was empty before
        if len(self.client_waiting_queues[seq_group.client_id]) == 1:
            # lift counter
            cnts = [v for k, v in self.served.items()
                      if (len(self.client_waiting_queues[k]) > 0 and k != seq_group.client_id)]
            if len(cnts) > 0:
                self.served[seq_group.client_id] = max(self.served[seq_group.client_id], min(cnts))

    # 被preempt的请求，被重新加入到对应client的waiting队列的头部
    # client_waiting_queues中不可能没有对应的client
    def appendleft(self, seq_group: SequenceGroup):
        self.client_waiting_queues[seq_group.client_id].appendleft(seq_group)

    # 被preempt的请求，被重新加入到对应client的waiting队列的头部
    # client_waiting_queues中不可能没有对应的client
    def extendleft(self, sg_list: List[SequenceGroup]):
        for seq_group in sg_list:
            self.client_waiting_queues[seq_group.client_id].appendleft(seq_group)


    # 请求被调度，或者请求不合法被赶出去
    def popleft(self):
        # 选出一个当前得分最少的client（credit-adjusted CPI）
        active_served = {k: v for k, v in self.served.items()}
        while True:
            client_id = min(active_served, key=active_served.get)
            if len(self.client_waiting_queues[client_id]) > 0:
                return self.client_waiting_queues[client_id].popleft()
            else:
                del active_served[client_id]
    
    # 为了应对waiting_queue[0]的访问 
    def __getitem__(self, index:int):
        assert index == 0
        # 选出一个当前得分最少的client
        active_served = {k: v for k, v in self.served.items()}
        while True:
            client_id = min(active_served, key=active_served.get)
            if len(self.client_waiting_queues[client_id]) > 0:
                return self.client_waiting_queues[client_id][0]
            else:
                del active_served[client_id]

    # 判断整个队列是否为空，依次查看每个client对应的队列是否为空
    def __bool__(self):
        if not self.client_waiting_queues:
            return False
        
        for client_id, waiting_queue in self.client_waiting_queues.items():
            if waiting_queue:
                return True
            
        return False
    
    def __len__(self):
        if not self.client_waiting_queues:
            return 0
        
        length = 0
        for client_id, waiting_queue in self.client_waiting_queues.items():
            if waiting_queue:
                length += len(waiting_queue)

        return length
    
    def get_client_queue_lengths(self) -> Dict[client_id_type, int]:
        """统计每个client对应waiting_queue的长度"""
        queue_lengths = {}
        for client_id, waiting_queue in self.client_waiting_queues.items():
            queue_lengths[client_id] = len(waiting_queue)
        return queue_lengths
    
    # 为了应对for语句的查找
    def __iter__(self):
        all_waiting_queue = deque()
        for client_id, waiting_queue in self.client_waiting_queues.items():
            all_waiting_queue.extend(waiting_queue)
        
        return iter(all_waiting_queue) 

    def remove(self, seq_group: SequenceGroup):
        for client_id, waiting_queue in self.client_waiting_queues.items():
            if seq_group in waiting_queue:
                waiting_queue.remove(seq_group)
                return
            
    # {client_id:(hit_length, not_hit_length)}
    # credits: Optional[Dict[client_id_type, float]] 给 prefix 创建者的补偿
    # 根据 prefill 结果更新计分板，使用 credit 抵扣后的净成本
    def update_prefill_served(
        self,
        prefill_served: Dict[client_id_type, List[float]],
        credits: Optional[Dict[client_id_type, float]] = None,
    ):
        if credits:
            for cid, cred in credits.items():
                self.credits[cid] = self.credits.get(cid, 0) + cred

        for client_id, prefill_hit_state in prefill_served.items():
            cost = self.hit_weight * prefill_hit_state[0] + self.prefill_weight * prefill_hit_state[1]
            D = self.credits.get(client_id, 0)
            net_cost = max(0.0, cost - D)
            self.served[client_id] = self.served.get(client_id, 0) + net_cost
            self.credits[client_id] = max(0.0, D - cost)

            if client_id not in self.served_requests:
                self.served_requests[client_id] = 0
                self.served_tokens[client_id] = 0
                self.hit_tokens_count[client_id] = 0
                self.miss_tokens_count[client_id] = 0
            self.served_requests[client_id] += 1
            self.served_tokens[client_id] += (prefill_hit_state[0] + prefill_hit_state[1])
            self.hit_tokens_count[client_id] += prefill_hit_state[0]
            self.miss_tokens_count[client_id] += prefill_hit_state[1]
        

class ReqQueue():

    def __init__(self, hit_weight=0, prefill_weight=0.1, decode_weight=0.2) -> None:
        # weight
        self.hit_weight = hit_weight
        self.prefill_weight = prefill_weight
        self.decode_weight =  decode_weight

        # 各用户CPI使用量记分板
        self.served: Dict[client_id_type, float] = {}

        # 各用户的队列
        self.waiting_queue: Deque[SequenceGroup] = deque()
        self.client_counter: Counter = Counter()

        # 各用户各调度了多少请求
        self.served_requests: Dict[client_id_type, int] = {}

        # 各用户各调度了多少token
        self.served_tokens: Dict[client_id_type, int] = {}

        # 各用户命中token数量
        self.hit_tokens_count: Dict[client_id_type, int] = {}

        # 各用户miss token数量
        self.miss_tokens_count:  Dict[client_id_type, int] = {}

    # 新到来的请求进入waiting队列，加在队列尾部
    def append(self, seq_group: SequenceGroup):
        # 请求入队
        # client刚出现，新建一个该client对应的deque
        if self.client_counter[seq_group.client_id] == 0:
            self.served[seq_group.client_id] = 0

        self.client_counter[seq_group.client_id] += 1
        self.waiting_queue.append(seq_group)

        # waiting queue of the client was empty before
        if self.client_counter[seq_group.client_id] == 1:
            # lift counter
            cnts = [v for k, v in self.served.items()
                      if (self.client_counter[k] > 0 and k != seq_group.client_id)]
            if len(cnts) > 0:
                self.served[seq_group.client_id] = max(self.served[seq_group.client_id], min(cnts))

        

    # 被preempt的请求，被重新加入到对应client的waiting队列的头部
    # client_waiting_queues中不可能没有对应的client
    def appendleft(self, seq_group: SequenceGroup):
        self.waiting_queue.appendleft(seq_group)
        self.client_counter[seq_group.client_id] += 1

    # 被preempt的请求，被重新加入到对应client的waiting队列的头部
    # client_waiting_queues中不可能没有对应的client
    def extendleft(self, sg_list: List[SequenceGroup]):
        for seq_group in sg_list:
            self.waiting_queue.appendleft(seq_group)
            self.client_counter[seq_group.client_id] += 1

    # 请求被调度，或者请求不合法被赶出去
    def popleft(self):
        request = self.waiting_queue.popleft()
        self.client_counter[request.client_id] -= 1
        if self.client_counter[request.client_id] == 0:
            del self.client_counter[request.client_id]
        return request
    
    # 为了应对waiting_queue[0]的访问 
    def __getitem__(self, index:int):
        return self.waiting_queue[index]

    # 判断整个队列是否为空，依次查看每个client对应的队列是否为空
    def __bool__(self):
        return bool(self.waiting_queue)
    
    def __len__(self):
        return len(self.waiting_queue)
    
    def get_client_queue_lengths(self) -> Dict[client_id_type, int]:
        """统计每个client对应waiting_queue的长度"""
        return dict(self.client_counter)
    
    # 为了应对for语句的查找
    def __iter__(self):     
        return iter(self.waiting_queue) 

    def remove(self, seq_group: SequenceGroup):
        self.waiting_queue.remove(seq_group)
        return
            
    # {client_id:(hit_length, not_hit_length)}
    # 根据prefill计算的返回结果，来更新计分板
    def update_prefill_served(
        self,
        prefill_served: Dict[client_id_type, List[float]],
        credits: Optional[Dict[client_id_type, float]] = None,
    ):
        for client_id, prefill_hit_state in prefill_served.items():
            self.served[client_id] = self.served.get(client_id, 0) + (
                self.hit_weight * prefill_hit_state[0] + self.prefill_weight * prefill_hit_state[1]
            )
            if client_id not in self.served_requests:
                self.served_requests[client_id] = 0
                self.served_tokens[client_id] = 0
                self.hit_tokens_count[client_id] = 0
                self.miss_tokens_count[client_id] = 0
            self.served_requests[client_id] += 1
            self.served_tokens[client_id] += (prefill_hit_state[0] + prefill_hit_state[1])
            self.hit_tokens_count[client_id] += prefill_hit_state[0]
            self.miss_tokens_count[client_id] += prefill_hit_state[1]
        
        # 仅在prefill之后输出各client被调度的请求数量
        timestamp = time.time()
        with open('/mnt/baizhuoyan/fair-caching/lmcache-tests/outputs/request_servied.json', 'a') as f:
            output_data = {"timestamp": timestamp, "requests": self.served_requests}
            json.dump(output_data, f, indent=4)
            f.write(',\n')  # 在换行前输出一个逗号
        
        # 在prefill之后输出各client的hit token数量
        with open('/mnt/baizhuoyan/fair-caching/lmcache-tests/outputs/hit_tokens_count.json', 'a') as f:
            output_data = {"timestamp": timestamp, "hit_tokens": self.hit_tokens_count}
            json.dump(output_data, f, indent=4)
            f.write(',\n')  # 在换行前输出一个逗号

        # 在prefill之后输出各client的miss token数量
        with open('/mnt/baizhuoyan/fair-caching/lmcache-tests/outputs/miss_tokens_count.json', 'a') as f:
            output_data = {"timestamp": timestamp, "miss_tokens": self.miss_tokens_count}
            json.dump(output_data, f, indent=4)
            f.write(',\n')  # 在换行前输出一个逗号

        # 仅在prefill之后输出各client的CPI计分板情况
        with open('/mnt/baizhuoyan/fair-caching/lmcache-tests/outputs/cpi_score_output.json', 'a') as f:
            output_data = {"timestamp": timestamp, "served": self.served}
            json.dump(output_data, f, indent=4)
            f.write(',\n') 

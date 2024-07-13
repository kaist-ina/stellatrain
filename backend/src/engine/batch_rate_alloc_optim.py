from collections import OrderedDict
import numpy as np
import random
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
from scipy.optimize import minimize
import pandas as pd
import math
import time
from skopt import load as load_res
from enum import Enum

GPU_INIT_WARMUP_THRESH = 4
GPU_POINT_RELIABLE_THRESH = 4
EFFECTIVE_AFTER_ITER = 3
WAIT_EFFECT_ITER = 4
WAIT_EFFECT_ITER_MAIN = 5


@dataclass
class InstElem():
    iter: int
    batch: np.ndarray
    comp_ratio: float

@dataclass
class GlobalInitConfig():
    num_node: int
    num_gpus: int
    param_counts: List[int]
    grad_fp16: bool
    grad_idx_u16: bool
    profile_path: str


class SpecialDeque():
    def __init__(self):
        self.container = {}
        self.momentum = 0.9
        self.rl = 0
    
    def append(self, batch_size: int, thpt: float):
        if batch_size in self.container:
            self.container[batch_size] = self.container[batch_size] * self.momentum + thpt * (1 - self.momentum)
        else:
            self.container[batch_size] = thpt
        self.rl += 1

    def to_list(self):
        return list(self.container.items())
    
    def num_collected_samples(self):
        return self.rl
    
    def __len__(self):
        return self.container.__len__()

class GpuThroughputEstimator:
    def __init__(self):
        self.alpha: int = 30
        self.beta: float = 200.
        self.history = SpecialDeque()

    @staticmethod
    def func(params: Tuple[float, float], x: int) -> float:
        alpha, beta = params
        return np.minimum((beta / alpha) * np.array(x).astype(float), beta)

    def get_throughput(self, batch_size: int) -> float:
        '''
        Returns throughput in item/s
        '''
        return GpuThroughputEstimator.func((self.alpha, self.beta), batch_size)

    
    def update(self, fwd_ms: float, bwd_ms: float, batch_size: int):
        '''
        Update the throughput estimator
        '''
        self.history.append(batch_size, batch_size * 1000 / (bwd_ms))

    
    def fit(self):

        if (len(self.history) < 2):
            return
        
        x_data, y_data = zip(*self.history.to_list())

        # Use L2 Loss
        def error_func(params):
            return np.sum((GpuThroughputEstimator.func(params, x_data) - y_data) ** 2)

        initial_guess = [x_data[-1], y_data[-1]]
        print(f"Initial guess {initial_guess}")

        result = minimize(error_func, initial_guess, method='Nelder-Mead', bounds=((16, None), (150, None)))
        a_opt, b_opt = result.x
        print(f"Optimized params: alpha = {a_opt}, beta = {b_opt}, {result.fun}")

        self.alpha = a_opt
        self.beta = b_opt

class OptimizerPhase(Enum):
    INIT_WARMUP = 0
    INIT_COLLECT_X = 1
    RUNNING = 2

class MainOptimizer:

    def __init__(self, config: GlobalInitConfig, cb_get_latest_iter: Callable[[], int]):
        self.num_gpus = num_gpus = config.num_gpus
        self.config = config
        self.param_counts: List[int] = config.param_counts
        print(f"Initializing optimizer with {num_gpus} GPUs")
        self.alphas = np.ones(num_gpus) * 10.
        self.betas = np.ones(num_gpus) * 200.
        self.max_batch_sizes = np.ones(num_gpus) * np.inf
        self.history = [SpecialDeque() for _ in range(num_gpus)]
        self.history_iter = [0 for _ in range(num_gpus)]
        self.instruction_history: OrderedDict[int, InstElem] = OrderedDict()
        self.instruction_history_log = None

        self.initialized = [False for _ in range(num_gpus)]
        self.init_batch_sizes = np.ones(num_gpus) * 0
        self.init_comp_ratio = 0
        self.estim_throughput = None

        self.network_gap_history = [None for _ in range(num_gpus)]
        self.comp_time_history = [None for _ in range(num_gpus)]
        self.prev_fun = 0

        self.cb_get_latest_iter = cb_get_latest_iter
        
        if config.profile_path is None:
            warnings.warn("Model profile is missing. set environment variable MODEL_PROFILE to a path to a file containing the model profile.")

        self.res = load_res(config.profile_path)
        print(f"Loaded model profile from {config.profile_path}")
        print(f"Model Profile : {self.res.metadata}")
        
        self.x_scale = self.res.metadata['max_batch_size']
        self.y_scale = self.res.metadata['bandwidth']

        # if 'verson' not in self.res.metadata or self.res.metadata['version'] != 'v2':
        #     warnings.warn("Model version mismatch. Expected v2")

        
        x_min, x_max, y_min, y_max = \
            self.res.space[0][1].low, self.res.space[0][1].high, self.res.space[1][1].low, self.res.space[1][1].high
        
        self.x_min, self.x_max, self.y_min, self.y_max = \
            x_min * self.x_scale, x_max * self.x_scale, y_min * self.y_scale, y_max * self.y_scale
        
        self.lambda_2 = 5
        self.prev_x = None
        self.prev_thpt = None
        self.main_iter = 0
        self.history_result = []

        self.phase: OptimizerPhase = OptimizerPhase.INIT_WARMUP

    @staticmethod
    def f(alpha, beta, x):
        return np.minimum((beta.astype(float) / alpha) * x.astype(float), beta)
    
    def h(self, x, y):
        x, y = x / self.x_scale, y / self.y_scale
        if np.isscalar(x) and np.isscalar(y):
            return self.res.models[-1].predict([[x, y]])[0] * self.y_scale
        return self.res.models[-1].predict(np.array([x,y]).transpose())
    
    def parameterized_f(self: 'MainOptimizer', x: np.ndarray):
        assert x.shape[-1] == self.num_gpus
        return self.f(self.alphas, self.betas, x)
    
    def target_sub_func_2(self: 'MainOptimizer', x: np.ndarray):
        # coefficient of variation
        return np.std(x / self.parameterized_f(x), axis=-1) / np.mean(x / self.parameterized_f(x), axis=-1)

    def target_sub_func_1(self: 'MainOptimizer', thpt: Tuple[np.ndarray, float], x: np.ndarray, conversion_factor: float):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        x = x.round()
        if np.isscalar(thpt):
            thpt = np.broadcast_to(thpt, x[..., 0].shape)
        thpt = thpt * conversion_factor

        upper = 1 + self.h(np.sum(x, axis=-1), thpt)*50
        k = np.argmax(x / self.parameterized_f(x), axis=-1)
        x_k = x[np.arange(x.shape[0]),k]
        slowest_gpu_time = x_k / self.f(self.alphas[k], self.betas[k], x_k)
        sum_batches = np.sum(x, axis=-1)
        lower = sum_batches / slowest_gpu_time
        return upper / lower

    def target_func(self: 'MainOptimizer', thpt: Tuple[np.ndarray, float], x: np.ndarray, conversion_factor: float):
        if x.ndim == 1: x = np.expand_dims(x, axis=0)
        x = x.round()
        return self.target_sub_func_1(thpt, x, conversion_factor) + self.lambda_2 * self.target_sub_func_2(x)

    def step_main(self: 'MainOptimizer', _iter: int, throughput: float, df: pd.DataFrame) -> Optional[InstElem]:
        df = df.copy()
        # prev_estim_throughput = self.estim_throughput
        if self.estim_throughput is None:
            self.estim_throughput = throughput
        else:
            
            print(df.drop(columns=['location', 'grad_acc', 'gpu']))
            client_ids = df.index.values.tolist()
            reduce_fraction = 0
            wait_times, comp_times = [], []
            for i, client_id in enumerate(client_ids):
                wait_times.append(df.loc[client_id, 'bwd_fwd_gap_ms'] + df.loc[client_id, 'intra_fwd_gap_ms'])
                comp_times.append(3. / 2. * df.loc[client_id, 'bwd_ms'])

            min_wait_time = min(wait_times)
            avg_comp_time = np.average(comp_times)

            reduce_fraction = max(reduce_fraction, min_wait_time / (min_wait_time + avg_comp_time))
            print(f"Wait time {min_wait_time} ms, Comp time {avg_comp_time} ms, reduce fraction {reduce_fraction}")
            
            if reduce_fraction < 0.05:
                reduce_fraction = 0

            throughput_prev = self.prev_thpt * 0.9 + throughput * 0.1
            if reduce_fraction == 0:
                throughput = throughput_prev * 1.1 #+ 250 * (self.config.num_node - 1)
            else:
                throughput = throughput_prev * max(0.75, (1 - reduce_fraction))
    
            print(f"Estim thpt {throughput_prev} -> {throughput} (+-{reduce_fraction}), actual {throughput}")


        '''
            *Necessity of Conversion Factor*
            - bayesian optimization was done with assumtion that (meta.bs, meta.compression) can achieve meta.bandwidth.
            - e.g. (256, 0.9) -> 1 (Gbps)
            - however, in our system, what happens if GPUs are slower than that so we can achieve 1 Gbps (1024, 0.9) -> 1 Gbps?
            - thus we need conversion factor here:
            - first, measure the bandwidth requirement of our environment with given cpr. e.g. (256, 0.9) -> x Gbps
            - if "x" < 1, this means GPUs are slower than expected. Bandwidth gets relatively loosen -- This equivalent to that we have a larger bandwidth.
            - if "x" > 1, this means GPUs are faster than expected. Bandwidth gets relatively less room -- This equivalent to that we have a smaller bandwidth.
            - Thus we define the conversion factor c_f = 1 / x
        '''
        if self.prev_x is None:
            prev_aggr_throughput = np.sum(self.f(self.alphas, self.betas, np.ones(self.num_gpus) * self.res.metadata['batch_size']))
        else:
            prev_aggr_throughput = np.sum(self.f(self.alphas, self.betas, self.prev_x / np.sum(self.prev_x) * self.res.metadata['batch_size']))

        print(f"prev_aggr_throughput {prev_aggr_throughput} iter/s")

        conversion_factor = 1000 / (
            self.calculate_goodput_mbps(
                self.res.metadata['compression'], 
                1500 * (3/2) * self.res.metadata['batch_size'] / prev_aggr_throughput
            ) * self.res.metadata['bandwidth'])
        print(conversion_factor)

        # c_throughput = throughput * conversion_factor
        # print(f"Converted throughput {c_throughput} Mbps")
        
        def binary_search_min_cond(low, high, cond, max_step=20):
            """
            Find the minimum x such that cond(x) is true.

            :param low: The lower bound of the search range.
            :param high: The upper bound of the search range.
            :param cond: A function that takes an integer x and returns a boolean.
            :return: The minimum x for which cond(x) is true, or None if no such x exists.
            """
            if low > high:
                return None

            while low < high and max_step > 0:
                mid = low + (high - low) / 2
                if cond(mid):
                    high = mid
                else:
                    low = mid
                max_step -= 1

            return low
        
        def calculate_goodput_mbps(compression_ratio: float, batch_size = None):
            # print(f"Compression ratio {compression_ratio}")
            # batch_size_sum = np.sum(df['batch_size'])
            if batch_size is None:
                batch_size = df['batch_size']
            estimated_fwd_bwd_time_ms = 3/2 * np.average(self.f(self.alphas, self.betas, df['batch_size']))
            # print(f"Estimated fwd bwd time {estimated_fwd_bwd_time_ms} ms. Batch size sum {batch_size_sum}")
            return self.calculate_goodput_mbps(compression_ratio, estimated_fwd_bwd_time_ms)
        min_compression = binary_search_min_cond(0, 0.999, lambda x : calculate_goodput_mbps(x) <= throughput)
        print(f"min_compression {min_compression}, throughput {throughput} Mbps, actually {calculate_goodput_mbps(min_compression)}")

        if self.main_iter % 100 == 0:
            min_v = float("inf")
            result = None
            optim_start_time = time.time()
            scales = [1, 2]
            for _iter_scale in range(32 if self.main_iter == 0 else len(scales)):
                if self.main_iter == 0:
                    initial_x = np.array([random.randint(1, self.max_batch_sizes[i]) for i in range(self.num_gpus)])
                else:
                    thpt_scale = np.clip(self.prev_thpt / np.clip(throughput ,0.001, None), 0.1, 16)
                    initial_x = np.clip(self.prev_x * thpt_scale * scales[_iter_scale], 1, self.max_batch_sizes).round()
                    print(initial_x, thpt_scale, conversion_factor)
                def func(x):
                    return self.target_func(throughput, x, conversion_factor)
                r = minimize(func, initial_x, bounds=[(1, self.max_batch_sizes[i]) for i in range(self.num_gpus)], 
                            method='Nelder-Mead', options={'adaptive': True})
                if r.fun < min_v:
                    min_v = r.fun
                    result = r
            optim_end_time = time.time()
            print(f"Optimization time {optim_end_time - optim_start_time} s")
            result_x, result_fun = result.x, result.fun
        else:
            print(f"Using previous result, {self.prev_x}")
            result_x = self.prev_x
            result_fun = self.prev_fun
            
        self.main_iter += 1
        self.prev_x = result_x
        self.prev_fun = result_fun
        self.prev_thpt = throughput
        self.history_result.append((throughput, result_x, result_fun))


        
        min_compression = binary_search_min_cond(0, 0.999, lambda x : calculate_goodput_mbps(x, result_x) <= throughput)
        print(f"min_compression {min_compression}, throughput {throughput} Mbps, actually {calculate_goodput_mbps(min_compression, result_x)}")
        min_compression = 1 - (1 - min_compression) / (self.config.num_node - 1)
        print(f"As we have {self.config.num_node} Nodes, setting min_compression {min_compression}")

        
        # req_mbps_nocomp = self.calculate_goodput_mbps(0, 1500 * (3/2) * np.sum(result.x) / np.sum(self.parameterized_f(result.x)))
        # print(req_mbps_nocomp)
        # new_comp = max(0, min(1 - throughput / req_mbps_nocomp, 0.999))
    
        new_inst = InstElem(_iter, result_x.round(), min_compression)
        return new_inst

    def update(self, timestamps: List[float], gpu_ids: List[int], dics: List[dict]):
        '''
        Update the throughput estimator
        '''
        last_instances = {}
        for timestamp, gpu_id, dic in zip(timestamps, gpu_ids, dics):
            batch_size = dic['batch_size']
            bwd_ms = dic['bwd_ms']
            comp_ratio = dic['comp_ratio']

            if self.history_iter[gpu_id] >= GPU_INIT_WARMUP_THRESH:
                self.history[gpu_id].append(batch_size, batch_size * 1000 / (bwd_ms))
            self.history_iter[gpu_id] += 1

            last_instances[gpu_id] = dic
        
        self.fit_alpha_betas()

        for gpu_id, dic in last_instances.items():
            batch_size = dic['batch_size']
            bwd_ms = dic['bwd_ms']
            comp_ratio = dic['comp_ratio']

            gap = dic['bwd_fwd_gap_ms'] + dic['intra_fwd_gap_ms']
            if self.network_gap_history[gpu_id] is None:
                self.network_gap_history[gpu_id] = gap
            else:
                self.network_gap_history[gpu_id] = self.network_gap_history[gpu_id] * 0.8 + gap * 0.1

            comp_time =  dic['bwd_ms']
            if self.comp_time_history[gpu_id] is None:
                self.comp_time_history[gpu_id] = comp_time
            else:
                self.comp_time_history[gpu_id] = self.comp_time_history[gpu_id] * 0.75 + comp_time * 0.25


            if all(x is not None for x in self.network_gap_history):
                nw_bound = min(self.network_gap_history)
                # print(f"Network bound {nw_bound} ms")


            if self.phase == OptimizerPhase.INIT_WARMUP:
                self.initialized[gpu_id] = True
                self.init_comp_ratio = comp_ratio
                self.init_batch_sizes[gpu_id] = batch_size

                if all(self.initialized) and 0 not in self.instruction_history:
                    print("Creating initial history")
                    print(f"GPU {gpu_id} initialized with batch size {batch_size} and comp ratio {comp_ratio}")
                    inst_elem = InstElem(0, self.init_batch_sizes, self.init_comp_ratio)
                    self.instruction_history[0] = inst_elem

                if all([h.num_collected_samples() >= GPU_INIT_WARMUP_THRESH for h in self.history]):
                    self.phase = OptimizerPhase.INIT_COLLECT_X
                    print("Phase 1 completed")

    def update_max_batch_size(self, gpu_id: int, max_batch_size: int):
        self.max_batch_sizes[gpu_id] = max_batch_size

    def fit_alpha_betas(self):
        for i in range(self.num_gpus):
            if (self.history[i].num_collected_samples() < GPU_POINT_RELIABLE_THRESH):
                continue
            
            x_data, y_data = zip(*self.history[i].to_list())
            # print(x_data, y_data)

            def func(params: Tuple[float, float], x: int) -> float:
                alpha, beta = params
                return np.minimum((beta / alpha) * np.array(x).astype(float), beta)

            # Use L2 Loss
            def error_func(params):
                return np.sum((func(params, x_data) - y_data) ** 2)

            initial_guess = [self.alphas[i], self.betas[i]] #[x_data[-1], y_data[-1]]
            initial_guess = np.clip(initial_guess, a_min = np.array([16, 100]) , a_max=None)
            result = minimize(error_func, initial_guess, method='Nelder-Mead', bounds=((16, None), (100, None)))
            a_opt, b_opt = result.x
            # print(f"Optimized params: alpha = {a_opt}, beta = {b_opt}, {result.fun}")

            self.alphas[i] = a_opt
            self.betas[i] = b_opt

    def step_init_collect_x(self: 'MainOptimizer', _iter: int, throughput: float, df: pd.DataFrame) -> Optional[InstElem]:

        assert self.max_batch_sizes.max() < np.inf

        # start with batch size of 4
        if len(self.instruction_history) == 0:
            return np.ones(self.num_gpus) * 4
        
        prev_inst: InstElem = next(reversed(self.instruction_history.values()))

        if np.all(prev_inst.batch == self.max_batch_sizes):
            print("Running phase")
            self.phase = OptimizerPhase.RUNNING

            for h in self.history:
                h.momentum = 0.999

            return self.step_main(_iter, throughput, df)

        print(prev_inst.batch, self.max_batch_sizes)
        new_inst_batch = np.round(np.min([prev_inst.batch * 1.5, self.max_batch_sizes], axis=0))
        print(new_inst_batch)
        new_inst = InstElem(_iter, new_inst_batch, self.init_comp_ratio)
        return new_inst
        
    def step(self: 'MainOptimizer', timestamp: float, _iter: int, throughput: float, df: pd.DataFrame) -> Optional[InstElem]:
        step_req_timestamp = timestamp
        result = None
        if len(self.instruction_history) > 0 and _iter:
            prev_inst_iter, prev_inst = next(reversed(self.instruction_history.items()))
            if _iter < prev_inst_iter + (WAIT_EFFECT_ITER_MAIN if self.phase == OptimizerPhase.RUNNING else WAIT_EFFECT_ITER):
                # print(f"Waiting for effect of previous instruction, {prev_inst_iter} -> {_iter}")
                return

        if self.phase == OptimizerPhase.INIT_COLLECT_X:
            result =  self.step_init_collect_x(_iter, throughput, df)
        elif self.phase == OptimizerPhase.RUNNING:
            result = self.step_main(_iter, throughput, df)

        if result is None:
            return

        result.iter = self.cb_get_latest_iter() + EFFECTIVE_AFTER_ITER
        self.instruction_history[result.iter] = result

        if self.instruction_history_log is None:
            self.instruction_history_log = open("instruction_history_log.txt", "w")
        self.instruction_history_log.write(f"{result.iter},{result.comp_ratio},{','.join([str(x) for x in result.batch.tolist()])}\n")
        self.instruction_history_log.flush()
        
        return result

    def update_scoreboard(self, df: pd.DataFrame):
        client_ids = df.index.values.tolist()
        min_gap_ms = float("inf")
        for i, client_id in enumerate(client_ids):
            df.loc[client_id, ('alpha', 'beta')] = (self.alphas[i], self.betas[i])

            x = df.loc[client_id, 'batch_size']
            if x is not None:
                min_gap_ms = min(min_gap_ms, df.loc[client_id, 'bwd_fwd_gap_ms'] + df.loc[client_id, 'intra_fwd_gap_ms'])
                expected_fwd_bwd_ms = 3. / 2. * self.f(self.alphas[i], self.betas[i], np.array([x]))[0]
                # print(f"Batch size {x} expected fwd bwd time {expected_fwd_bwd_ms} ms, actual fwd bwd time {df.loc[client_id, 'fwd_ms'] + df.loc[client_id, 'bwd_ms']} ms")
                df.loc[client_id, 'req_bw'] = \
                    self.calculate_goodput_mbps(df.loc[client_id, 'comp_ratio'],
                                                3. / 2. * self.f(self.alphas[i], self.betas[i], np.array([x]))[0])

    def estimate_tx_bytes(self, compression_ratio: float):
        assert compression_ratio < 1
        assert compression_ratio >= 0

        MTU = 1538
        tcp_header_overheads = 0x42
        MSS = MTU - tcp_header_overheads

        total_param_bytes = 0 

        for param_count in self.config.param_counts:
            idx_bytes_per_elem = 2 if self.config.grad_idx_u16 and param_count < 65536 else 4
            val_bytes_per_elem = 2 if self.config.grad_fp16 else 4
            param_key_len = 16 # estimate
            compressed_param_count = int(max(1, math.ceil((1 - compression_ratio) * param_count)))
            zmq_payload_len = 2 + 1 + 8 + param_key_len + 1 + 1 + 1 + 8 + idx_bytes_per_elem * compressed_param_count + 1 + 8 + val_bytes_per_elem * compressed_param_count
            num_segments = math.ceil((zmq_payload_len + MSS - 1) / (MSS))
            total_len = zmq_payload_len + num_segments * tcp_header_overheads
            total_param_bytes += total_len

        return total_param_bytes


    def calculate_goodput_mbps(self, compression_ratio: float, fwd_bwd_time_ms: float):

        bytes = self.estimate_tx_bytes(compression_ratio)
        # print(f"Estimated bytes {bytes} for compression ratio {compression_ratio} and fwd bwd time {fwd_bwd_time_ms} ms")
        return bytes * 8 / (fwd_bwd_time_ms * 1000)
    

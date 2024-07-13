import os, sys
from typing import Any, List, Optional, Tuple
import zmq
import pandas as pd
import math
from collections import defaultdict, deque
from batch_rate_alloc_optim import MainOptimizer, GpuThroughputEstimator, GlobalInitConfig
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../build/src/proto' ))
from message_pb2 import Request, Response, Instruction
from threading import Thread
from queue import Queue, Empty
import time

debug_out = open('/tmp/fasterdp-batch-rate-alloc.log', 'w')    

MAX_BATCH_SIZE_MAP = {
    'RTX 4090': 90,
    'RTX 3090': 90,
    'RTX 2080 Ti': 36,
    'A100-SXM4-40GB': 120,
    'V100-SXM2-32G': 112
}
class BatchSizeRateAlloc:

    def __init__(self):
        self.context = zmq.Context()
        self.scoreboard = pd.DataFrame(
            columns=['client_id', 'location', 'gpu', 'uiter', 'batch_size', 'grad_acc', 'comp_ratio', 'alpha', 'beta', 'tx_thpt', 'rx_thpt', 'req_bw', 'fwd_ms', 'bwd_ms', 'bwd_fwd_gap_ms', 'intra_fwd_gap_ms']
        )
        self.scoreboard.set_index('client_id', inplace=True)
        self.scoreboard['uiter'].astype("Int64")
        self.uiter_min = None
        self.uiter_max = None
        self.next_instruct_iter = None
        self.comp_ratio = None
        self.throughput_approximator = defaultdict(GpuThroughputEstimator)
        self.message_queue = defaultdict(deque)
        self.main_optim: MainOptimizer = None
        self.lateset_iter = defaultdict(int)
        self.is_finished = False
        

        self.num_node: int = 0
        self.num_gpus: int = 0
        self.param_counts: List[int] = []
        self.grad_fp16: bool = False
        self.grad_idx_u16: bool = False


        self.rx_q: Queue[Optional[Tuple[float, int, Response]]] = Queue()
        self.tx_q: defaultdict[int, Queue[Optional[Response]]] = defaultdict(Queue)
        self.io_thread = Thread(target=self.io_thread_main)
        self.io_thread.start()

        self.alpha_beta_q: Queue[Any] = Queue()
        self.alpha_beta_thread = Thread(target=self.alpha_beta_thread_main)
        self.alpha_beta_thread.start()
        
        self.logic_q: Queue[Any] = Queue()
        self.logic_thread = Thread(target=self.logic_thread_main)
        self.logic_thread.start()

        self.disabled = os.getenv('DISABLE_RATE_ADAPT', '0') != '0' or os.getenv('MODEL_PROFILE', '') == ''
        self.is_finished = False
        self.client_id_sender_mapping = {}
        self.sender_client_id_mapping = {}

        while True:
            obj = self.rx_q.get()
            if obj is None:
                break

            timestamp, sender, req = obj

            
            if req.type == Request.Type.INIT:
                print(f"Starting Batch Rate Allocation server")
            elif req.type == Request.Type.GLOBAL_TELEMETRY_INIT:
                self.global_telemetry_init(req)
            elif req.type == Request.Type.LOCAL_TELEMETRY_INIT:
                self.init_telemetry(req)
            elif req.type == Request.Type.LOCAL_TELEMETRY:
                # update_start_time = time.time()
                self.client_id_sender_mapping[req.telemetry.client_id] = sender
                self.sender_client_id_mapping[sender] = req.telemetry.client_id
                self.update_telemetry(timestamp, req)
                # print(f"Update time: {time.time() - update_start_time}, rx_q_sz = {self.rx_q.qsize()}")
            else:
                print("Unknown request type")

        print("Terminating Batch Rate Allocation server")

    def __del__(self):
        self.is_finished = True
        self.alpha_beta_q.put(None)

        try:
            self.io_thread.join()
            self.alpha_beta_thread.join()
        except:
            pass
        
        self.socket.close()
        self.context.term()
        
    def send(self, client_id, msg: Response):
        self.socket.send_multipart([client_id, b'', msg.SerializeToString()])

    def global_telemetry_init(self, req):
        self.num_node = req.global_telemetry_init.num_nodes
        self.num_gpus = req.global_telemetry_init.num_gpus
        self.grad_fp16 = req.global_telemetry_init.grad_fp16
        self.grad_idx_u16 = req.global_telemetry_init.grad_idx_u16
        self.param_counts = req.global_telemetry_init.param_counts
        
        print(f"Global Telemetry Init: num_nodes = {self.num_node}, num_gpus = {self.num_gpus}, grad_fp16 = {self.grad_fp16}, grad_idx_u16 = {self.grad_idx_u16}, param_counts = Total {sum(self.param_counts):,}")

        config = GlobalInitConfig(self.num_node, self.num_gpus, self.param_counts, self.grad_fp16, self.grad_idx_u16, os.getenv('MODEL_PROFILE'))
        self.main_optim = MainOptimizer(config, self.get_latest_iter)

    def update_telemetry(self, timestamp, req):
        if req.telemetry.client_id not in self.scoreboard.index:
            print(f"Client {req.telemetry.client_id} not found in scoreboard")
            return
        
        def ewma(old, new):
            if old is None:
                return new
            alpha = 0.1
            return old * alpha + new * (1 - alpha)
        
        dic = {
            'uiter': int(req.telemetry.uiter_count),
            'batch_size': req.telemetry.config.ubatch_size,
            'grad_acc': req.telemetry.config.gradient_accumulation,
            'comp_ratio': req.telemetry.config.compression_rate,
            'tx_thpt': ewma(self.scoreboard.loc[req.telemetry.client_id, 'tx_thpt'], req.telemetry.measurements.grad_send_thpt_mbps),
            'rx_thpt': ewma(self.scoreboard.loc[req.telemetry.client_id, 'rx_thpt'], req.telemetry.measurements.grad_recv_thpt_mbps),
            'fwd_ms': req.telemetry.measurements.fwd_delay_ms,
            'bwd_ms': req.telemetry.measurements.bwd_delay_ms,
            'bwd_fwd_gap_ms': req.telemetry.measurements.bwd_fwd_gap_ms,
            'intra_fwd_gap_ms': req.telemetry.measurements.intra_fwd_gap_ms
        }

        self.scoreboard.loc[req.telemetry.client_id].update(dic)
        
        gpu_id = self.scoreboard.index.values.tolist().index(req.telemetry.client_id)
        gpu_name = self.scoreboard.loc[req.telemetry.client_id, 'gpu']
        assert gpu_id >= 0
        assert self.main_optim is not None
        assert gpu_name in MAX_BATCH_SIZE_MAP

        self.main_optim.update_max_batch_size(gpu_id, MAX_BATCH_SIZE_MAP[gpu_name])
        self.alpha_beta_q.put((timestamp, gpu_id, dic))

        # print(f"Client {req.telemetry.client_id} updated scoreboard, uiter = {dic['uiter']}")


        
    def init_telemetry(self, req):
        gpu_name = req.local_telemetry_init.gpu_name.replace('NVIDIA', '').replace('GeForce', '').strip()

        self.scoreboard.loc[req.local_telemetry_init.client_id] = {
            'location': f"{req.local_telemetry_init.rank}.{req.local_telemetry_init.local_rank}",
            'gpu': gpu_name,
            'uiter': None,
            'batch_size': None,
            'grad_acc': None,
            'comp_ratio': None,
            'tx_thpt': None,
            'rx_thpt': None,
            'req_bw': None,
            'alpha': None,
            'beta': None,
            'fwd_ms': None,
            'bwd_ms': None,
            'bwd_fwd_gap_ms': None,
            'intra_fwd_gap_ms': None,
        }
        self.scoreboard.sort_index(inplace=True)

    def calculate_goodput_mbps(self, compression_ratio: float, fwd_bwd_time_ms: float):
        assert compression_ratio < 1
        assert compression_ratio >= 0
        
        param_bytes = 0
        for param_count in self.param_counts:
            param_bytes += max(1, math.ceil((1 - compression_ratio) * param_count)) \
                            * ((2 if self.grad_fp16 else 4) + (2 if self.grad_idx_u16 and param_count < 65536 else 4))

        return param_bytes * 8 * (self.num_node - 1) / (fwd_bwd_time_ms * 1000)
        

    def logic(self, timestamp: float):
        if self.main_optim:
            valid_thpt_measurments = [x for x in self.scoreboard['rx_thpt'].values.tolist() if x is not None and x > 0]
            if len(valid_thpt_measurments) == 0:
                thpt = 0
            else:
                thpt = min(valid_thpt_measurments)
            result = self.main_optim.step(timestamp, self.uiter_min, thpt, self.scoreboard)
            if result is not None:
                print(result)
                instructions = []
                sorted_cid = list(sorted(cid for cid in self.scoreboard.index.values))
                for i, cid in enumerate(sorted_cid):
                    instruction = Instruction()
                    instruction.valid = True
                    instruction.client_id = cid
                    instruction.effective_since_uiter = int(result.iter)
                    instruction.config.ubatch_size = int(result.batch[i])
                    instruction.config.gradient_accumulation = 1
                    instruction.config.compression_rate = result.comp_ratio
                    instructions.append(instruction)
                return instructions

    def get_latest_iter(self):
        return max(self.lateset_iter.values())
    
    def alpha_beta_thread_main(self: 'BatchSizeRateAlloc'):
        while not self.is_finished:
            obj = self.alpha_beta_q.get()
            if obj is None:
                return
            lst_obj = [obj]
            while True:
                try:
                    obj = self.alpha_beta_q.get_nowait()
                    if obj is None: break
                    lst_obj.append(obj)
                except Empty:
                    break

            # print(f"Handling {len(lst_obj)} objects")
            timestamps, gpu_ids, dics = zip(*lst_obj)
            self.main_optim.update(timestamps, gpu_ids, dics)
            self.main_optim.update_scoreboard(self.scoreboard)

            if self.scoreboard['uiter'].count() != self.scoreboard['uiter'].notna().sum():
                continue
            
            new_uiter_min = self.scoreboard['uiter'].min()
            self.uiter_max = self.scoreboard['uiter'].max()

            if self.comp_ratio is None:
                self.comp_ratio = dics[-1]['comp_ratio']

            # if new iteration is detected
            if self.uiter_min is None or new_uiter_min > self.uiter_min:
                self.uiter_min = new_uiter_min
                self.logic_q.put(timestamps[-1])


    def logic_thread_main(self: 'BatchSizeRateAlloc'):
        while not self.is_finished:
            obj = self.alpha_beta_q.get()
            if obj is None:
                return
            while True:
                try:
                    obj = self.alpha_beta_q.get_nowait()
                except Empty:
                    break
            assert obj is not None

            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
                print(self.scoreboard, file=debug_out)
                
            assignment = self.logic(obj)
            if assignment is not None:
                print(self.scoreboard.drop(columns=['location', 'grad_acc']))
                if self.disabled:
                    return
                for instruction in assignment:
                    msg = Response()
                    msg.type = Response.Type.INSTRUCTION
                    msg.instruction.CopyFrom(instruction)
                    # print(f"Sending instruction to {instruction.client_id} for iter {instruction.effective_since_uiter}")
                    sender = self.client_id_sender_mapping[instruction.client_id]
                    self.tx_q[sender].put(msg)


    def io_thread_main(self: 'BatchSizeRateAlloc'):

        BIND_URI = "tcp://*:5556"
        print(f"Starting server on {BIND_URI}")
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(BIND_URI)

        while True:
            recv_env = self.socket.recv_multipart()
            recv_timestamp = time.time()
            req = Request()
            sender = recv_env[0]
            req.ParseFromString(recv_env[2])
            if req.type == Request.Type.LOCAL_TELEMETRY:
                self.lateset_iter[req.telemetry.client_id] = int(req.telemetry.uiter_count)

            if req.type == Request.Type.TERMINATE:
                self.rx_q.put(None)
                msg = Response()
                self.send(sender, msg)
                break

            self.rx_q.put((recv_timestamp, sender, req))

            if sender in self.sender_client_id_mapping:
                client_id = self.sender_client_id_mapping[sender]
            else:
                client_id = None

            try:
                msg = self.tx_q[sender].get_nowait()
                # print(f"[SEND] instruction to {client_id} for iter {msg.instruction.effective_since_uiter}")
                    
            except Empty:
                msg = Response()
                msg.type = Response.Type.OK    
                # print(f"[SEND] Null        to {client_id} for iter {msg.instruction.effective_since_uiter}")

            self.send(sender, msg)

if __name__ == "__main__":
    try:
        with open('/tmp/fasterdp-batch-rate-alloc.pid', 'r') as f:
            os.killpg(int(f.read().strip()), 9)
    except:
        pass

    with open('/tmp/fasterdp-batch-rate-alloc.pid', 'w') as f:
        f.write(str(os.getpgid(os.getpid())))

    BatchSizeRateAlloc()
    
    os.unlink('/tmp/fasterdp-batch-rate-alloc.pid')

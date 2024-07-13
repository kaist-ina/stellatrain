

from collections import OrderedDict
from random import shuffle
from typing import Any, Callable, Union
import torch
import os
import sys
from shutil import copyfile

def traverse_tensor_sum(obj, fn: Callable[[torch.Tensor], Any]) -> int:
    def _internal_traverse(o):
        if isinstance(o, list):
            return sum(_internal_traverse(c) for c in o)
        if isinstance(o, OrderedDict):
            return sum(_internal_traverse(v) for v in o.values())
        if isinstance(o, dict):
            return sum(_internal_traverse(v) for v in o.values())
        if isinstance(o, tuple):
            return sum(_internal_traverse(c) for c in o)
        if isinstance(o, torch.Tensor):
            return fn(o)
        if isinstance(o, zip):
            return sum(_internal_traverse(c) for c in o)
        return 0
    return _internal_traverse(obj)

def get_bytes(obj: Any) -> int: 
    def _get_bytes(tensor: torch.Tensor) -> int:
        raw = tensor.nelement() * tensor.element_size()
        byte_align = 512
        return raw + (byte_align - raw) % byte_align
    return traverse_tensor_sum(obj, _get_bytes)

def get_logical_bytes(t: torch.Tensor) -> int:
    return t.nelement() * t.element_size()

def get_bytes_recursive(t: torch.Tensor) -> int:
    counted_data_ptr = set()
    estimated_x_size = 0

    def get_exact_bytes_fn(fn) -> int:
        if not fn:
            return 0

        tensor_attributes = [attr for attr in dir(fn) if '_saved_' in attr and torch.is_tensor(getattr(fn, attr))]
        result = 0
        for attr in tensor_attributes:
            tensor: torch.Tensor = getattr(fn, attr)
            if tensor.storage().data_ptr() not in counted_data_ptr:
                counted_data_ptr.add(tensor.storage().data_ptr())
                b1 = get_bytes(tensor)
                
                consider1 = not any(attr.endswith(x) for x in ['running_mean', 'running_var', 'weight'])
                consider2 = consider1

                if consider1:
                    result += b1
                if attr.endswith('self'):
                    nonlocal estimated_x_size
                    estimated_x_size = b1
                    
                b2 = get_exact_bytes_fn(tensor.grad_fn)
                if consider2:
                    result += b2
            else:
                b1 = get_bytes(tensor)
                
        for next_fn, _ in fn.next_functions:
            result += get_exact_bytes_fn(next_fn)

        return result

    result = get_bytes(t)
    counted_data_ptr.add(t.storage().data_ptr())
    result += get_exact_bytes_fn(t.grad_fn)
    result -= estimated_x_size

    return result


def get_shape(obj):
  def _internal_shape(o):
    if isinstance(o, list):
      return list(_internal_shape(c) for c in o)
    if isinstance(o, OrderedDict):
      return OrderedDict((k, _internal_shape(v)) for k, v in o.items())
    if isinstance(o, dict):
      return dict((k, _internal_shape(v)) for k, v in o.items())
    if isinstance(o, tuple):
      return tuple(_internal_shape(c) for c in o)
    if isinstance(o, torch.Tensor):
      return o.shape
    if isinstance(o, zip):
      return tuple(_internal_shape(c) for c in o)
    return o
  return _internal_shape(obj)



class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val   = 0
		self.avg   = 0
		self.sum   = 0
		self.count = 0

	def update(self, val, n=1):
		self.val   = val
		self.sum   += val * n
		self.count += n
		self.avg   = self.sum / self.count

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred    = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].flatten().float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res



def copy_script_file(writer, file_name):
    model_checkpoints_folder = os.path.join(writer.log_dir)
    try:
        os.makedirs(model_checkpoints_folder, exist_ok=True)
        copyfile(file_name, os.path.join(model_checkpoints_folder, os.path.basename(file_name)))
        with open(os.path.join(model_checkpoints_folder, "cmdline"), "w") as f:
            f.write(" ".join(sys.argv))

    except:
        pass


from inspect import isfunction
from argparse import Namespace
import json

def log_current_config(writer, args: Namespace):
    model_checkpoints_folder = os.path.join(writer.log_dir)
    args_to_write = {k: v for k, v in vars(args).items() if not isfunction(v)}
    os.makedirs(model_checkpoints_folder, exist_ok=True)
    json_out = json.dumps(args_to_write, indent=4)
    print(json_out)
    with open(os.path.join(model_checkpoints_folder, "config.json"), "w") as f:
        f.write(json_out)



class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

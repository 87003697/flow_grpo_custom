import numpy as np
import os
import torch
import torch.distributed as dist
import torch.autograd as autograd


def get_model(model):
    """获取模型实例（处理 DataParallel 包装）"""
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


def is_dist_avail_and_initialized():
    """检查分布式训练是否可用且已初始化"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """获取世界大小（分布式训练的进程数）"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """获取当前进程的排名"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """检查是否为主进程"""
    return get_rank() == 0


def scaled_all_reduce(tensors, is_scale=True):
    """执行缩放的 all_reduce 操作"""
    world_size = get_world_size()
    # 单进程情况下不需要归约
    if world_size == 1:
        return tensors
    # 排队归约操作
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # 等待归约完成
    for reduction in reductions:
        reduction.wait()
    # 缩放结果
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def all_gather_batch(tensors):
    """对提供的张量执行 all_gather 操作"""
    world_size = get_world_size()
    # 单进程情况下不需要归约
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # 性能优化
        )
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor
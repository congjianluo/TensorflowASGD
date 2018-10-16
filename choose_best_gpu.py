#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ay27 at 2018/4/8
import os


def set_best_gpu(top_k=1):
    best_gpu = _scan(top_k)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, best_gpu))
    return best_gpu


def _scan(top_k):
    CMD1 = 'nvidia-smi| grep MiB | grep -v Default | cut -c 4-8'
    CMD2 = 'nvidia-smi -L | wc -l'
    CMD3 = 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits'

    total_gpu = int(os.popen(CMD2).read())

    assert top_k <= total_gpu, 'top_k > total_gpu !'

    # first choose the free gpus
    gpu_usage = set(map(lambda x: int(x), os.popen(CMD1).read().split()))
    free_gpus = set(range(total_gpu)) - gpu_usage

    # then choose the most memory free gpus
    gpu_free_mem = list(map(lambda x: int(x), os.popen(CMD3).read().split()))
    gpu_sorted = list(sorted(range(total_gpu), key=lambda x: gpu_free_mem[x], reverse=True))[len(free_gpus):]

    res = list(free_gpus) + list(gpu_sorted)
    return res[:top_k]


if __name__ == '__main__':
    print(set_best_gpu(4))
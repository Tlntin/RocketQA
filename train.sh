#!/bin/bash

echo "测试多卡运行，已经可以正常使用了"

export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
# 预分配显存大小的比例 80%
export FLAGS_fraction_of_gpu_memory_to_use=0.80
export TRANSLATOR_VERBOSITY=3
#export CUDA_VISIBLE_DEVICES=0,1,2,3

export GLOG_v=1
python3 -m paddle.distributed.launch --gpus=0,1 train.py
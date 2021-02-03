#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}
DEFAULT_PORT=28500
MASTER_PORT=$(($2 + $DEFAULT_PORT))

CUDA_VISIBLE_DEVICES=$2 $PYTHON -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 $(dirname "$0")/train.py $1 --launcher pytorch ${@:3}


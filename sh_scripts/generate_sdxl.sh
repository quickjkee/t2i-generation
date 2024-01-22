#!/bin/bash

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1

python3 -m torch.distributed.run --master-addr=0.0.0.0:1206 --nproc_per_node=2 generate.py \
	--dataset custom \
	--steps 50 \
	--save_path samples \
	--w 8 \
	--name stable_xl \
	--bs 12 \
	--model xl \
	--seeds 0,1,2,3 \
	--max_cnt 6234

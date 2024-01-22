#!/bin/bash

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1

python3 -m torch.distributed.run --master-addr=0.0.0.0:1205 --nproc_per_node=2 generate.py \
	--dataset custom \
	--local_model_path pretrained/sd-v2-1 \
	--steps 50 \
	--scheduler DDIM \
	--save_path samples \
	--w 8 \
	--name v2-1 \
	--bs 50 \
	--model v2-1 \
	--seeds 0,1,2,3 \
	--max_cnt 6234

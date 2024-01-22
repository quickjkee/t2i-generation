#!/bin/bash

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,5,7

python -m torch.distributed.run --standalone --master-addr=0.0.0.0:1205 --nproc_per_node=4 generate.py \
	--dataset custom \
	--local_model_path pretrained/sd-v1-5 \
	--steps 50 \
	--save_path samples \
	--w 8 \
	--name v1-5 \
	--bs 50 \
	--model v1-5 \
	--seeds 0,1,2,3 \
	--max_cnt 6234

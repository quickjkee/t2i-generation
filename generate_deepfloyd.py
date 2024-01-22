# make sure you're logged in with \`huggingface-cli login\`
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers import DDIMScheduler, DDPMScheduler, HeunDiscreteScheduler, EulerDiscreteScheduler
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

import torch
import os
import pandas as pd
import argparse
import dist_util as dist
import numpy as np
import tqdm
from nirvana_utils import copy_out_to_snapshot
import time
import cv2
from PIL import Image
from models import MODELS


parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--steps', type=int, default=50, help='number of inference steps during sampling')
parser.add_argument('--w', type=float, default=8.0)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--max_cnt', type=int, default=5000, help='number of maximum generated samples')
parser.add_argument('--save_path', type=str, default='./generated_images')
parser.add_argument('--scheduler', type=str, default=None)
parser.add_argument('--dataset', type=str, default='custom')
parser.add_argument('--name', type=str, default=None)

parser.add_argument('--generate_seed', type=int, default=6)
parser.add_argument('--seeds', type=str, default="")
parser.add_argument('--resize_output', type=int, default=None)

parser.add_argument('--model', type=str, default='DeepFloyd')
parser.add_argument('--local_model_path', type=str, default=None)

args = parser.parse_args()

dist.init()
dist.print0('Args:')
for k, v in sorted(vars(args).items()):
    dist.print0('\t{}: {}'.format(k, v))

###################
# Prepare prompts #
###################
if args.dataset == 'coco':
    df = pd.read_csv('./prompts/coco.csv')
    all_text = list(df['caption'])
elif args.dataset == 'laion':
    df = pd.read_csv('./prompts/laion.csv')
    all_text = list(df['caption'])
elif args.dataset == 'ir':
    df = pd.read_csv('./prompts/image_reward_db.csv')
    all_text = list(df['caption'])
    assert len(all_text) == args.max_cnt == 463
elif args.dataset == 'yagen':
    df = pd.read_csv('./prompts/yagen_test.csv')
    all_text = list(df['prompt_en'])
    assert len(all_text) == args.max_cnt == 300
elif args.dataset == 'custom':
    df = pd.read_csv('./prompts/custom.tsv', sep='\t')
    all_text = list(df['prompt'])
    assert len(all_text) == 6234

all_text = all_text[: args.max_cnt]
num_batches = ((len(all_text) - 1) // (args.bs * dist.get_world_size()) + 1) * dist.get_world_size()
all_batches = np.array_split(np.array(all_text), num_batches)
rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]

index_list = np.arange(len(all_text))
all_batches_index = np.array_split(index_list, num_batches)
rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]

##################
# Load T2I model #
##################

stage_1, stage_2, stage_3 = MODELS["DeepFloyd"](
    local_path=args.local_model_path
)

dist.print0("Default stage 1 scheduler:")
dist.print0(stage_1.scheduler)

dist.print0("Default stage 2 scheduler:")
dist.print0(stage_2.scheduler)

dist.print0("Default stage 1 scheduler:")
dist.print0(stage_3.scheduler)

if dist.get_rank() == 0:
    t0 = time.time()

if len(args.seeds) > 0:
    seeds = [int(seed) for seed in args.seeds.split(',')]
else:
    seeds = [args.generate_seed]

##########################
# Setup custom scheduler #
##########################

for seed in seeds:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    ##### setup save configuration #######
    if args.name is None:
        save_dir = os.path.join(args.save_path,
                                f'steps_{args.steps}_w_{args.w}_seed_{seed}')
    else:
        save_dir = os.path.join(args.save_path,
                                f'steps_{args.steps}_w_{args.w}_seed_{seed}_name_{args.name}')

    dist.print0("save images to {}".format(save_dir))

    if dist.get_rank() == 0:
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

    ## generate images ##
    for cnt, mini_batch in enumerate(tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
        text = list(mini_batch)
        prompt_embeds, negative_embeds = stage_1.encode_prompt(text)

        # stage 1
        image = stage_1(
            prompt_embeds=prompt_embeds, 
            negative_prompt_embeds=negative_embeds, 
            generator=generator, output_type="pt"
        ).images

        # stage 2
        image = stage_2(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            output_type="pt",
        ).images

        # stage 3
        images = stage_3(
            prompt=text, 
            image=image, 
            noise_level=100, 
            generator=generator
        ).images

        for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
            if args.resize_output is not None:
                images[text_idx] = Image.fromarray(cv2.resize(
                     np.array(images[text_idx]), 
                    (args.resize_output, args.resize_output),
                    interpolation=cv2.INTER_AREA
                ))
            images[text_idx].save(os.path.join(save_dir, f'{global_idx}.jpg'))
# Done.
dist.barrier()
if dist.get_rank() == 0:
    print(f"Overall time: {time.time()-t0:.3f}")
    d = {'caption': all_text}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(args.save_path, 'generated_prompts.csv'))
    copy_out_to_snapshot(args.save_path)
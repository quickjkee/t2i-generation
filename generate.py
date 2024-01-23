# make sure you're logged in with \`huggingface-cli login\`
import copy

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers import DDIMScheduler, DDPMScheduler, HeunDiscreteScheduler, EulerDiscreteScheduler
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

import torch
torch.set_num_threads(10)
import os
import pandas as pd
import argparse
import dist_util as dist
import numpy as np
import random
import tqdm
from nirvana_utils import copy_out_to_snapshot
import time
import cv2
from PIL import Image

from utils import get_prompts, get_t2i_model
from models import MODELS
import yt.wrapper as yt
from yt.wrapper import YtClient

def image2bytes(
    image, img_ext: str = ".png"
) -> bytes:
    """
    image: C x H x W
    """
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, image = cv2.imencode(img_ext, image)
    return image.tobytes()

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

parser.add_argument('--model', type=str, default='v1-5')
parser.add_argument('--local_model_path', type=str, default=None)
parser.add_argument('--local_refiner_path', type=str, default=None)

args = parser.parse_args()

dist.init()
dist.print0('Args:')
for k, v in sorted(vars(args).items()):
    dist.print0('\t{}: {}'.format(k, v))

###################
# Prepare prompts and model
###################

all_text = get_prompts(args)
#random.shuffle(all_text)
all_text = all_text[: args.max_cnt]
pipe, refiner = get_t2i_model(args)

# Splitting
num_batches = ((len(all_text) - 1) // (args.bs * dist.get_world_size()) + 1) * dist.get_world_size()
all_batches = np.array_split(np.array(all_text), num_batches)
rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]

index_list = np.arange(len(all_text))
all_batches_index = np.array_split(index_list, num_batches)
rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]

# Seeds
if dist.get_rank() == 0:
    t0 = time.time()

if len(args.seeds) > 0:
    seeds = [int(seed) for seed in args.seeds.split(',')]
else:
    seeds = [args.generate_seed]

##########################
# Setup custom scheduler #
##########################

## generate images ##
client = YtClient(proxy="hahn")
path = yt.TablePath("//home/yr/quickjkee/sdxl_coco", append=True, client=client)
row = [{'prompt': None, 'seeds': None, 'model': None, 'prompt_source': None,
         'image_1': None, 'image_2': None, 'image_3': None, 'image_4': None, 'image_5': None,
         'image_6': None, 'image_7': None, 'image_8': None, 'image_9': None, 'image_10': None}
        ]

save_dir = args.save_path
if dist.get_rank() == 0:
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
dist.barrier()

files = os.listdir(save_dir)
if len(files) > 0:
    first_el = [int(file.split('_')[0]) for file in files if int(file.split('_')[4]) == dist.get_rank()]
    skip_iter = max(first_el)
    print(f'Generation starting from {skip_iter}')
else:
    skip_iter = -10

print(rank_batches_index)
for cnt, mini_batch in enumerate(tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
    mini_batch_idx = rank_batches_index[cnt][0]
    if mini_batch_idx <= skip_iter:
        continue
    text = list(mini_batch)
    new_row = copy.deepcopy(row)

    prompt = text[0]
    prompt = prompt.replace('/', ' ')
    name_old = f'{mini_batch_idx}_{prompt}_{args.name}_{args.dataset}_{dist.get_rank()}'
    print(mini_batch_idx, dist.get_rank())
    #new_row[0]['prompt'] = text[0]
    #new_row[0]['model'] = args.name
    #new_row[0]['prompt_source'] = args.dataset

    for it, seed in enumerate(range(mini_batch_idx * 10, mini_batch_idx * 10 + 10)):
        #new_row[0]['seeds'] = list(range(mini_batch_idx * 10, mini_batch_idx * 10 + 10))
        name = f'{name_old}_{seed}.jpg'
        generator = torch.Generator().manual_seed(seed)
        image = pipe(
            text,
            generator=generator,
            num_inference_steps=args.steps,
            guidance_scale=args.w,
        ).images[0].resize((512, 512))
        #image = image2bytes(image)
        #new_row[0][f'{image}_{it+1}'] = image
        image.save(os.path.join(save_dir, name))

    if dist.get_rank() == 0:
        if cnt % 50 == 0:
            copy_out_to_snapshot(args.save_path)
    #client.write_table(path, new_row, raw=False)

# Done.
dist.barrier()
if dist.get_rank() == 0:
    print(f"Overall time: {time.time() - t0:.3f}")
    d = {'caption': all_text}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(args.save_path, 'generated_prompts.csv'))
    copy_out_to_snapshot(args.save_path)

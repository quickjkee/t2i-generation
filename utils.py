from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers import DDIMScheduler, DDPMScheduler, HeunDiscreteScheduler, EulerDiscreteScheduler
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

import pandas as pd
import dist_util as dist
from models import MODELS


def get_prompts(args):
    if args.dataset == 'coco':
        df = pd.read_csv('./prompts/coco.csv')
        all_text = list(df['caption'])
    elif args.dataset == 'cocotrain':
        df = pd.read_csv('./prompts/train2014_50k.csv')
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
    elif args.dataset == 'pickscore':
        df = pd.read_csv('./prompts/pickscore_40k.csv')
        all_text = list(df['prompt_en'])
        assert len(all_text) == args.max_cnt == 300
    elif args.dataset == 'custom':
        df = pd.read_csv('./prompts/custom.tsv', sep='\t')
        all_text = list(df['prompt'])
        assert len(all_text) == 6234

    return all_text


def get_t2i_model(args):
    refiner = None
    if args.model == 'xl':
        pipe = MODELS["stabilityai/stable-diffusion-xl-base-1.0"](
            local_path=args.local_model_path
        )
    elif args.model == 'xl_refiner':
        pipe, refiner = MODELS["stabilityai/stable-diffusion-xl-refiner-1.0"](
            local_path=args.local_model_path,
            local_refiner_path=args.local_refiner_path,
            use_refiner=True
        )
    elif args.model == 'v2-1':
        pipe = MODELS["stabilityai/stable-diffusion-2-1"](
            local_path=args.local_model_path
        )
    elif args.model == 'v1-5':
        pipe = MODELS["runwayml/stable-diffusion-v1-5"](
            local_path=args.local_model_path
        )
    elif args.model == 'addxl':
        pipe = MODELS["stabilityai/sdxl-turbo"](
            local_path=args.local_model_path
        )
    elif args.model == 'lcmxl':
        pipe = MODELS["latent-consistency/lcm-sdxl"](
            local_path=args.local_model_path
        )
    else:
        raise Exception(f"Not found {args.model}")

    dist.print0("Default scheduler:")
    dist.print0(pipe.scheduler)

    ##########################
    # Setup custom scheduler #
    ##########################

    if args.scheduler:
        if args.scheduler == 'DDPM':
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        elif args.scheduler == 'DDIM':
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif args.scheduler == 'Heun':
            pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
        elif args.scheduler == 'DPMSolver':
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif args.scheduler == 'ODE':
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.use_karras_sigmas = False
        else:
            raise NotImplementedError

        dist.print0("New scheduler:")
        dist.print0(pipe.scheduler)

    return pipe, refiner
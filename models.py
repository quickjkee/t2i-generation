import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableUnCLIPPipeline, DiffusionPipeline
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, AutoPipelineForText2Image, UNet2DConditionModel, LCMScheduler


def get_sdxl(local_path=None, local_refiner_path=None, use_refiner=False, use_compile=True):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        local_path if local_path else "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, variant="fp16",
        add_watermarker=False,
        use_safetensors=True
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.requires_safety_checker = False
    pipe.safety_checker = None
    pipe.to("cuda")
    if use_compile:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    if use_refiner:
        refiner = DiffusionPipeline.from_pretrained(
            local_refiner_path if local_refiner_path else "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            force_zeros_for_empty_prompt=True
        )
        refiner.to("cuda")
        if use_compile:
            refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
        return pipe, refiner
    else:
        return pipe


def get_sd_v2(local_path=None, use_compile=True):
    pipe = StableDiffusionPipeline.from_pretrained(
        local_path if local_path else "stabilityai/stable-diffusion-2-1", 
        torch_dtype=torch.float16, variant="fp16"
    )
    if use_compile:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.requires_safety_checker = False
    pipe.safety_checker = None
    return pipe.to("cuda")


def get_sd_v1(local_path=None, use_compile=True):
    pipe = StableDiffusionPipeline.from_pretrained(
        local_path if local_path else "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16, variant="fp16"
    )
    if use_compile:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.requires_safety_checker = False
    pipe.safety_checker = None
    return pipe.to("cuda")


def get_unclip_small():
    from diffusers import UnCLIPScheduler, DDPMScheduler
    from diffusers.models import PriorTransformer
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection

    prior_model_id = "kakaobrain/karlo-v1-alpha"
    data_type = torch.float16
    prior = PriorTransformer.from_pretrained(prior_model_id, subfolder="prior", torch_dtype=data_type)

    prior_text_model_id = "openai/clip-vit-large-patch14"
    prior_tokenizer = CLIPTokenizer.from_pretrained(prior_text_model_id)
    prior_text_model = CLIPTextModelWithProjection.from_pretrained(prior_text_model_id, torch_dtype=data_type)
    prior_scheduler = UnCLIPScheduler.from_pretrained(prior_model_id, subfolder="prior_scheduler")
    prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)

    stable_unclip_model_id = "stabilityai/stable-diffusion-2-1-unclip-small"

    pipe = StableUnCLIPPipeline.from_pretrained(
        stable_unclip_model_id,
        torch_dtype=data_type,
        variant="fp16",
        prior_tokenizer=prior_tokenizer,
        prior_text_encoder=prior_text_model,
        prior=prior,
        prior_scheduler=prior_scheduler,
    )
    pipe.requires_safety_checker = False
    pipe.safety_checker = None
    return pipe.to("cuda")

def get_addxl(local_path=None):
    pipe_turbo = AutoPipelineForText2Image.from_pretrained(local_path if local_path else "stabilityai/sdxl-turbo",
                                                           torch_dtype=torch.float16,
                                                           variant="fp16")
    pipe_turbo.to("cuda")
    return pipe_turbo

def get_lcmxl(local_path=None):
    unet = UNet2DConditionModel.from_pretrained(local_path if local_path else "latent-consistency/lcm-sdxl",
                                                torch_dtype=torch.float16,
                                                variant="fp16")
    pipe_distill = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet,
                                                     torch_dtype=torch.float16)

    pipe_distill.scheduler = LCMScheduler.from_config(pipe_distill.scheduler.config)
    pipe_distill.to("cuda")
    return pipe_distill


def get_unclip():
    pipe = StableUnCLIPPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    return pipe.to("cuda")


def get_deepfloyd():
    # stage 1
    stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
    # stage_1.enable_model_cpu_offload()
    stage_1.requires_safety_checker = False
    stage_1.safety_checker = None

    # stage 2
    stage_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
    )
    # stage_2.enable_model_cpu_offload()

    # stage 3
    safety_modules = {
        "feature_extractor": stage_1.feature_extractor,
        "safety_checker": stage_1.safety_checker,
        "watermarker": stage_1.watermarker,
    }
    stage_3 = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
    )
    stage_3.enable_model_cpu_offload()

    stage_1 = stage_1.to("cuda")
    stage_2 = stage_2.to("cuda")
    return stage_1, stage_2, stage_3


MODELS = {
    "stabilityai/stable-diffusion-xl-base-1.0": get_sdxl,
    "stabilityai/stable-diffusion-xl-refiner-1.0": get_sdxl,
    "stabilityai/stable-diffusion-2-1": get_sd_v2,
    "runwayml/stable-diffusion-v1-5": get_sd_v1,
    "stabilityai/stable-diffusion-2-1-unclip": get_unclip,
    "stabilityai/stable-diffusion-2-1-unclip-small": get_unclip_small,
    'stabilityai/sdxl-turbo': get_addxl,
    'latent-consistency/lcm-sdxl': get_lcmxl,
    "DeepFloyd": get_deepfloyd,
}
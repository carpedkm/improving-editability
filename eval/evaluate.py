#!/usr/bin/env python
from __future__ import annotations
import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import numpy as np
import json
import torch
from typing import Optional
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import argparse
from diffusion.sa_solver_diffuses import SASolverScheduler
from diffusers import DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from scripts.pipeline_pixart_inpaint_with_latent_memory_improved import PixArtAlphaInpaintLMPipeline
from evaluate_functions import evaluate_clip_layerwise_score, evaluate_llava_layerwise_score
from multiedit_dataset import multiedit_DATASET

class ImageStore:
    def __init__(self, image: Optional[Image.Image] = None):
        self.image = image

    def get_image(self) -> Optional[Image.Image]:
        return self.image

    def store_image(self, image: Image.Image) -> None:
        self.image = image

def set_scheduler(pipe, scheduler_type: str, dpms_guidance_scale: float, sas_guidance_scale: float) -> float:
    if scheduler_type == 'DPM-Solver':
        pipe.scheduler = DPMSolverMultistepScheduler()
        guidance_scale = dpms_guidance_scale
    elif scheduler_type == 'SA-Solver':
        pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config, algorithm_type='data_prediction')
        guidance_scale = sas_guidance_scale
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    return guidance_scale

def run_pipeline(pipe, args, prompt: str, image: Optional[Image.Image] = None, mask_image: Optional[Image.Image] = None, 
                 inpaint: bool = False, cattn_masking: bool = False, multi_query_disentanglement: bool = False, 
                 seed: int = 334) -> Image.Image:
    latent_memory = args.latent_memory if image is not None else None
    generator = torch.Generator(device=f'cuda:{args.GPU_IDX}').manual_seed(seed)
    guidance_scale = set_scheduler(pipe, args.scheduler_type, args.dpms_guidance_scale, args.sas_guidance_scale)
    
    result_tensor, latent_memory_new = pipe(
        prompt=prompt,
        image=image if inpaint else None,
        mask_image=mask_image,
        strength=1.0,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.batch_size,
        inpaint=inpaint,
        latent_memory=latent_memory,
        vanilla_ratio=args.vanilla_ratio,
        cattn_masking=cattn_masking,
        multi_query_disentanglement=multi_query_disentanglement
    )

    result_tensor = result_tensor.images
    if latent_memory is None:
        args.latent_memory = latent_memory_new
    else:
        args.prev_latent_memory = args.latent_memory
        args.latent_memory = latent_memory_new

    return result_tensor[0]

def main():
    parser = argparse.ArgumentParser(description="PixArt Inpainting Evaluation")
    parser.add_argument('--model_version', type=str, default='PixArtAlpha', help='Model version')
    parser.add_argument('--model_base', type=str, default='InpaintLM', help='Model base')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--resolution', type=int, default=1024, help='Resolution')
    parser.add_argument('--dpms_guidance_scale', type=float, default=7.5, help='DPM-Solver guidance scale')
    parser.add_argument('--sas_guidance_scale', type=float, default=3.0, help='SA-Solver guidance scale')
    parser.add_argument('--num_inference_steps', type=int, default=20, help='Number of inference steps')
    parser.add_argument('--prompt', type=str, default='', help='Prompt')
    parser.add_argument('--negative_prompt', type=str, 
                       default='deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, '
                               'extra limb, missing limb, floating limbs, mutated hands and fingers, '
                               'disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW', 
                       help='Negative prompt')
    parser.add_argument('--scheduler_type', type=str, default='DPM-Solver', help='Scheduler type')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--result_dir', type=str, default='./output/ours', 
                       help='Result directory')
    parser.add_argument('--exp_name', type=str, default='PixArtInpaintExp', help='Experiment name')
    parser.add_argument('--vanilla_ratio', type=float, default=0.2, help='Vanilla ratio')
    parser.add_argument('--dataset', type=str, 
                       default='path-to-json', 
                       help='Path to the dataset')
    parser.add_argument('--cattn_masking', action='store_true', default=False, help='Use Cross-Attention masking')
    parser.add_argument('--multi_query_disentanglement', action='store_true', default=False, 
                       help='Use multi-query disentanglement')
    parser.add_argument('--seed', type=int, default=334, help='Random seed')
    parser.add_argument('--shard', type=int, default=0, help='Shard index')

    args = parser.parse_args()
    args.latent_memory = None
    args.prev_latent_memory = None

    image_storage = ImageStore(None)

    os.makedirs(f"{args.result_dir}/{args.exp_name}", exist_ok=True)
    OmegaConf.save(config=vars(args), f=f"{args.result_dir}/{args.exp_name}/configs.yaml")

    pipe_inpaint = PixArtAlphaInpaintLMPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        torch_dtype=torch.float16,
        use_auth_token=False,
        local_files_only=True
    ).to(f'cuda:{args.gpu}')

    try:
        with open(args.dataset, 'r') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Error loading dataset {args.dataset}: {e}")
        sys.exit(1)

    dataset = {k: dataset[k] for k in list(dataset.keys())[:100]}
    dataset = {k: dataset[k] for k in list(dataset.keys())[25*args.shard:25*(args.shard+1)]}
    
    dataset = multiedit_DATASET(dataset, in_pipeline=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, data_ in tqdm(enumerate(val_loader), total=len(val_loader), desc="Generating images"):
        idx = idx + args.shard * 25
        to_save_prompt = ''
        save_dir = os.path.join(args.result_dir, 'gen', f'IDX{idx}')
        os.makedirs(save_dir, exist_ok=True)

        for layer in range(data_['total_layers']):
            mask = Image.fromarray(data_['bboxes'][0, layer, :, :].numpy().squeeze()).resize((args.resolution, args.resolution)).convert('L')
            background_caption = data_['background_prompt'][0]
            foreground_caption = data_['local_prompts'][layer][0]

            if layer == 0:
                res = run_pipeline(pipe_inpaint, args, background_caption, image=None, mask_image=None, 
                                 inpaint=False, cattn_masking=False, multi_query_disentanglement=False, seed=args.seed)
                res.save(os.path.join(save_dir, f'IDX{idx}_layer_background.png'))
                res.resize((128, 128)).save(os.path.join(save_dir, f'IDX{idx}_layer_background_lowres.png'))
                image_storage.store_image(res)
                init_image = image_storage.get_image()
                res = run_pipeline(pipe_inpaint, args, foreground_caption, image=init_image, mask_image=mask, 
                                 inpaint=True, cattn_masking=args.cattn_masking, 
                                 multi_query_disentanglement=args.multi_query_disentanglement, seed=args.seed)
                to_save_prompt += f'ID:{idx}\nbackground_prompt:{background_caption}\nLayer0:{foreground_caption}\n'
                image_storage.store_image(res)
            else:
                init_image = image_storage.get_image()
                res = run_pipeline(pipe_inpaint, args, foreground_caption, image=init_image, mask_image=mask, 
                                 inpaint=True, cattn_masking=args.cattn_masking, 
                                 multi_query_disentanglement=args.multi_query_disentanglement, seed=args.seed)
                image_storage.store_image(res)
                to_save_prompt += f'Layer{layer}:{foreground_caption}\n'
                
            res.save(os.path.join(save_dir, f'IDX{idx}_layer{layer}.png'))
            res.resize((128, 128)).save(os.path.join(save_dir, f'IDX{idx}_layer{layer}_lowres.png'))
        
        pipe_inpaint.layerstore._empty_all_layers()
        with open(os.path.join(save_dir, f'IDX{idx}_prompt.txt'), 'w') as f:
            f.write(to_save_prompt)

    val_loader.dataset.in_pipeline = False
    clip_score_c, clip_score_p, std_c, std_p = evaluate_clip_layerwise_score(
        os.path.join(args.result_dir, 'gen'), val_loader, args, dataset)
    print(f"Clip score (class): {clip_score_c:.4f}")
    print(f"Clip score std-dev (class): {std_c:.4f}")
    print(f"Clip score (prompt): {clip_score_p:.4f}")
    print(f"Clip score std-dev (prompt): {std_p:.4f}")
    
    bleu, cider, meteor, rouge, meteor_std = evaluate_llava_layerwise_score(
        os.path.join(args.result_dir, 'gen'), val_loader, args, dataset)
    print(f"BLEU 1/2/3/4 score: {bleu[1]:.4f} / {bleu[2]:.4f} / {bleu[3]:.4f} / {bleu[4]:.4f}")
    print(f"CIDEr score: {cider:.4f}")
    print(f"METEOR score: {meteor:.4f}")
    print(f"METEOR std-dev: {meteor_std:.4f}")
    print(f"ROUGE score: {rouge:.4f}")

if __name__ == '__main__':
    main()
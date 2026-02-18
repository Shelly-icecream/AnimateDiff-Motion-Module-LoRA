import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess
import json
import numpy as np
import glob
import cv2

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.utils.data import Dataset

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.data.dataset import WebVid10M
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, zero_rank_print

from tqdm import tqdm


def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
        
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank
class LoRALinear(nn.Module):
    def __init__(self, layer, rank=8, alpha=8):
        super().__init__()
        self.layer = layer
        self.scale = alpha / rank

        in_f = layer.in_features
        out_f = layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 冻结原权重
        for p in self.layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.layer(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale
def is_motion_lora_target(name, module):

    # 必须是 motion module
    if "motion_modules" not in name:
        return False

    if not any(block in name for block in ["attention_blocks.0", "attention_blocks.1"]):
        return False

    # ⭐ 只保留 attention projection
    if not any(k in name for k in ["to_q", "to_k", "to_v", "to_out"]):
        return False

    # ⭐ 只接受 Linear
    if not isinstance(module, torch.nn.Linear):
        return False

    return True
def inject_motion_lora(unet, rank=8, alpha=8):

    print("\n🚀 Injecting Motion LoRA (SAFE MODE)...", flush=True)

    count = 0

    for name, module in list(unet.named_modules()):

        if not is_motion_lora_target(name, module):
            continue

        parent_name = ".".join(name.split(".")[:-1])
        child_name  = name.split(".")[-1]

        parent = unet.get_submodule(parent_name)

        setattr(
            parent,
            child_name,
            LoRALinear(module, rank=rank, alpha=alpha)
        )

        print("✔ Motion LoRA:", name)
        count += 1

    print(f"\n✅ Motion LoRA injection complete. Total = {count}")
    return unet
def downsample_gray(frame_bgr, scale=0.5):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        h, w = gray.shape
        gray = cv2.resize(gray, (max(1, int(w*scale)), max(1, int(h*scale))))
    return gray
EPS = 1e-6
def compute_slowmo_score(video_path, fps_sample=30, min_frames=12, scale=0.5):
    """
    返回: (score_norm, fps)
    score_norm: 0~1，越大越慢动作
    """

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps < 1:
        cap.release()
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < min_frames:
        cap.release()
        return None

    step = max(1, int(round(fps / fps_sample)))

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return None

    prev_gray = downsample_gray(prev, scale=scale)

    mags = []
    accel = []

    last_mag = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cur_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cur_idx % step != 0:
            continue

        gray = downsample_gray(frame, scale=scale)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        m = float(np.mean(mag))
        mags.append(m)

        if last_mag is not None:
            accel.append(abs(m - last_mag))
        last_mag = m

        prev_gray = gray

    cap.release()

    if len(mags) < 5:
        return None

    mags = np.array(mags, dtype=np.float32)
    accel = np.array(accel, dtype=np.float32) if len(accel) > 0 else np.array([0.0])

    mean_flow = float(np.mean(mags))
    mean_accel = float(np.mean(accel))

    # -----------------------------
    # 归一化（更稳定）
    # -----------------------------
    flow_norm = mean_flow / (mean_flow + 1.0)
    accel_norm = mean_accel / (mean_accel + 0.5)

    S1 = 1.0 - flow_norm
    S2 = 1.0 - accel_norm

    # fps bonus（轻权重）
    if fps >= 120:
        S3 = 1.0
    elif fps >= 60:
        S3 = 0.5
    else:
        S3 = 0.0

    score = 0.6 * S1 + 0.3 * S2 + 0.1 * S3
    score = float(np.clip(score, 0.0, 1.0))

    return score, fps
class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir,
        n_frames=8,
        size=256,
        stride=1,
        debug_stride_print=False,

        # ===== slowmo 相关 =====
        enable_slowmo_sampling=True,
        slowmo_fps_sample=30,
        slowmo_min_score=0.35,   # 过滤阈值：建议 0.30~0.45
        slowmo_cache=True
    ):
        self.root_dir = root_dir
        self.n_frames = n_frames
        self.size = size
        self.stride = stride
        self.debug_stride_print = debug_stride_print

        self.enable_slowmo_sampling = enable_slowmo_sampling
        self.slowmo_fps_sample = slowmo_fps_sample
        self.slowmo_min_score = slowmo_min_score
        self.slowmo_cache = slowmo_cache

        self.video_paths = glob.glob(os.path.join(root_dir, "**", "*.mp4"), recursive=True)

        print(f"🔥 Found {len(self.video_paths)} videos")
        print(f"🔥 Dataset config: n_frames={self.n_frames}, size={self.size}, stride={self.stride}")

        # ===== 预计算 slowmo_score（强烈推荐）=====
        self.slowmo_scores = None
        self.valid_paths = self.video_paths

        if self.enable_slowmo_sampling:
            self._build_slowmo_cache()

    def _build_slowmo_cache(self):
        print("🧠 Computing slowmo scores (one-time)...")

        scores = []
        paths = []

        # 新增：保存 slowmo token 映射
        self.slowmo_tags = {}

        for p in tqdm(self.video_paths):
            out = compute_slowmo_score(
                p,
                fps_sample=self.slowmo_fps_sample
            )

            if out is None:
                continue

            score, fps = out

            # 只保留慢动作视频
            if score < self.slowmo_min_score:
                continue

            paths.append(p)
            scores.append(score)

            # ✅ 关键：给通过的视频绑定 slowmo token
            self.slowmo_tags[p] = "<slowmo>"

        if len(paths) == 0:
            print("⚠️ No clips passed slowmo filter. Fallback to full dataset.")
            self.valid_paths = self.video_paths
            self.slowmo_scores = None
            self.slowmo_tags = {}
            return

        self.valid_paths = paths
        self.slowmo_scores = np.array(scores, dtype=np.float32)

        print(f"✅ Slowmo filter kept {len(self.valid_paths)} / {len(self.video_paths)} clips")
        print(f"✅ slowmo score stats: "
              f"min={self.slowmo_scores.min():.3f}, "
              f"mean={self.slowmo_scores.mean():.3f}, "
              f"max={self.slowmo_scores.max():.3f}")

    def __len__(self):
        return len(self.valid_paths)

    def _sample_index(self):
        if (not self.enable_slowmo_sampling) or (self.slowmo_scores is None):
            return random.randint(0, len(self.valid_paths) - 1)

        # 让高 slowmo_score 更容易被抽到
        # 温度系数：越大越偏向高分
        temp = 3.0
        w = np.power(self.slowmo_scores + 1e-3, temp)
        w = w / w.sum()
        return int(np.random.choice(len(self.valid_paths), p=w))

    def __getitem__(self, idx):
        for _ in range(10):
            # idx 由外部传入，但我们重采样
            real_idx = self._sample_index()
            path = self.valid_paths[real_idx]

            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            need = (self.n_frames - 1) * self.stride + 1
            if total < need or total <= 0:
                cap.release()
                continue

            start = random.randint(0, total - need)

            frames = []
            frame_indices = []

            ok = True
            for i in range(self.n_frames):
                frame_idx = start + i * self.stride
                frame_indices.append(frame_idx)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    ok = False
                    break

                frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
                frame = frame[:, :, ::-1]  # BGR -> RGB
                frames.append(frame)

            cap.release()

            if not ok or len(frames) != self.n_frames:
                continue

            if self.debug_stride_print and random.random() < 0.02:
                print(f"🔥 STRIDE CHECK: stride={self.stride}, total={total}, start={start}")
                print(f"🔥 selected frame idx = {frame_indices}")

            frames = np.stack(frames, axis=0)  # [F,H,W,3]
            frames = torch.from_numpy(frames).float() / 255.0
            frames = frames.permute(0, 3, 1, 2).contiguous()  # [F,3,H,W]

            txt_path = path.replace(".mp4", ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    base_prompt = f.read().strip()
            else:
                base_prompt = ""

            # 如果启用 slowmo 并且该视频被判定为慢动作
            if self.enable_slowmo_sampling and path in self.slowmo_tags:
                prompt = "<slowmo> " + base_prompt
            else:
                prompt = base_prompt

            return {"pixel_values": frames, "text": prompt}

        raise RuntimeError("Failed to fetch a valid video sample after 10 retries.")
def extract_motion_lora_state_dict(unet):
    # unet 可能是 DDP 包裹的
    if hasattr(unet, "module"):
        unet = unet.module

    sd = unet.state_dict()
    lora_sd = {}

    for k, v in sd.items():
        if "lora_A" in k or "lora_B" in k:
            lora_sd[k] = v.cpu()

    return lora_sd


def main(
    image_finetune: bool,
    
    name: str,
    use_wandb: bool,
    launcher: str,
    
    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)
    
    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    if not image_finetune:
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path, subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
        print("good")
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
        print("bad")
        
    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path: zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path

        m, u = unet.load_state_dict(state_dict, strict=False)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        #assert len(u) == 0
        print("Unexpected keys:", u[:50])
        print("Unexpected keys total:", len(u))

    """# ===============================
    # DEBUG: 打印 UNet 所有模块名称
    # ===============================
    print("\n================ UNET MODULES ================\n")

    for name, module in unet.named_modules():
        print(name)

    print("\n============= END UNET MODULES =============\n")"""

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    zero_rank_print("Injecting LoRA to motion modules...")

    unet = inject_motion_lora(unet, rank=8)
    #print("=== LoRA parameters after injection ===")
    #for name, _ in unet.named_parameters():
        #if "lora" in name.lower():
            #print(name)
    #count = sum(1 for n, _ in unet.named_parameters() if "lora" in n.lower())
    #print("Total LoRA param tensors:", count)

    # Set unet trainable parameters
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params)}")
        zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)



    # Load local video clips dataset
    train_dataset = VideoDataset(
        root_dir="/home/xixiangtang/AnimateDiff/clips",
        n_frames=train_data.n_frames,
        size=getattr(train_data, "sample_size", 256),
        stride=getattr(train_data, "sample_stride", 1),
        debug_stride_print=True
    )

    # Distributed sampler
    if num_processes > 1:
        distributed_sampler = DistributedSampler(
            train_dataset,
            num_replicas=num_processes,
            rank=global_rank,
            shuffle=True,
            seed=global_seed,
        )
    else:
        distributed_sampler = None

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=(distributed_sampler is None),
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers >0),
    )
    sample = next(iter(train_dataloader))

    if isinstance(sample, dict):
        video = sample["pixel_values"]
    else:
        video = sample

    print("🔥 VIDEO SHAPE =", video.shape)


    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = AnimationPipeline(
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
        ).to("cuda")
    else:
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
        )
    validation_pipeline.enable_vae_slicing()

    # DDP warpper
    unet.to(local_rank)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        unet.train()
        
        for step, batch in enumerate(train_dataloader):
            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]
                
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.gif", rescale=True)
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.png")
                    
            ### >>>> Training >>>> ###
            
            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]
                
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1):
                save_path = os.path.join(output_dir, "checkpoints")
                os.makedirs(save_path, exist_ok=True)

                lora_sd = extract_motion_lora_state_dict(unet)

                save_obj = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": lora_sd,  # 只保存 LoRA
                }

                if step == len(train_dataloader) - 1:
                    ckpt_name = f"motionlora-epoch-{epoch + 1}.ckpt"
                else:
                    ckpt_name = f"motionlora-step-{global_step}.ckpt"

                torch.save(save_obj, os.path.join(save_path, ckpt_name))
                logging.info(f"Saved MotionLoRA checkpoint to {save_path}/{ckpt_name} (global_step: {global_step})")

            # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []
                
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)
                
                height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                prompts = validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) else validation_data.prompts

                for idx, prompt in enumerate(prompts):
                    if not image_finetune:
                        sample = validation_pipeline(
                            prompt,
                            generator    = generator,
                            video_length = train_data.sample_n_frames,
                            height       = height,
                            width        = width,
                            **validation_data,
                        ).videos
                        save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                        samples.append(sample)
                        
                    else:
                        sample = validation_pipeline(
                            prompt,
                            generator           = generator,
                            height              = height,
                            width               = width,
                            num_inference_steps = validation_data.get("num_inference_steps", 25),
                            guidance_scale      = validation_data.get("guidance_scale", 8.),
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        samples.append(sample)
                
                if not image_finetune:
                    samples = torch.concat(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                    save_videos_grid(samples, save_path)
                    
                else:
                    samples = torch.stack(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.png"
                    torchvision.utils.save_image(samples, save_path, nrow=4)

                logging.info(f"Saved samples to {save_path}")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",    action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)

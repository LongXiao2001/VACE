# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
import gc
import math
import time
import random
import types
import logging
import traceback
from contextlib import contextmanager
from functools import partial

from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

from wan.text2video import (WanT2V, T5EncoderModel, WanVAE, shard_model, FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps, FlowUniPCMultistepScheduler)
from .modules.model import VaceWanModel
from .lora_utils import LoRAManager
from .runtime_vram import ModuleVRAMManager
from .layer_vram import (
    AutoWrappedLinear,
    AutoWrappedModule,
    enable_layer_vram_management,
    wrapped_model_offload,
    wrapped_model_onload,
)
from ..utils.preprocessor import VaceVideoProcessor


class WanVace(WanT2V):
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating VaceWanModel from {checkpoint_dir}")
        self.model = VaceWanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward,
                                                            usp_dit_forward_vace)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            for block in self.model.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.model.forward_vace = types.MethodType(usp_dit_forward_vace, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

        self.vid_proc = VaceVideoProcessor(downsample=tuple([x * y for x, y in zip(config.vae_stride, self.patch_size)]),
            min_area=480 * 832,
            max_area=480 * 832,
            min_fps=self.config.sample_fps,
            max_fps=self.config.sample_fps,
            zero_start=True,
            seq_len=32760,
            keep_last=True)
        self.lora_manager = None
        self.vram_manager = None
        self.memory_log_enabled = False
        self.memory_log_prefix = "WanVace"
        self.layer_vram_enabled = False

    def configure_runtime(
        self,
        block_swap=False,
        vace_hint_cpu_offload=False,
        lora_path=None,
        lora_scale=1.0,
        lora_dyn_offload=True,
        module_vram_management=False,
        module_vram_empty_cache=True,
        module_vram_verbose=False,
        module_vram_keep_resident=False,
        memory_log=False,
        memory_log_prefix="WanVace",
        layer_vram_management=False,
        num_persistent_param_in_dit=None,
        layer_vram_verbose=False,
    ):
        self.memory_log_enabled = memory_log
        self.memory_log_prefix = memory_log_prefix
        self.layer_vram_enabled = layer_vram_management
        if layer_vram_management:
            block_swap = False
            module_vram_management = False
            self._enable_layer_vram_management(
                num_persistent_param_in_dit=num_persistent_param_in_dit,
                verbose=layer_vram_verbose,
            )
        self.model.enable_block_swap(enabled=block_swap, device=self.device)
        self.model.enable_vace_hint_cpu_offload(enabled=vace_hint_cpu_offload)
        self.vram_manager = ModuleVRAMManager(
            device=self.device,
            enabled=module_vram_management,
            empty_cache=module_vram_empty_cache,
            verbose=module_vram_verbose,
        )
        self.vram_manager.set_logger(logging.info)
        self._register_vram_modules()
        self.model.set_module_loader(self._load_runtime_modules, keep_resident=module_vram_keep_resident)
        if lora_path:
            self.lora_manager = LoRAManager(
                model=self.model,
                lora_path=lora_path,
                scale=lora_scale,
                dynamic_offload=lora_dyn_offload,
            )
        else:
            self.lora_manager = None

    def _collect_named_module_types(self, root_module, names):
        classes = []
        for module in root_module.modules():
            if type(module).__name__ in names:
                cls = type(module)
                if cls not in classes:
                    classes.append(cls)
        return classes

    def _enable_layer_vram_management(self, num_persistent_param_in_dit=None, verbose=False):
        text_dtype = next(iter(self.text_encoder.model.parameters())).dtype
        text_module_map = {
            nn.Linear: AutoWrappedLinear,
            nn.Embedding: AutoWrappedModule,
        }
        for cls in self._collect_named_module_types(self.text_encoder.model, {"T5RelativeEmbedding", "T5LayerNorm"}):
            text_module_map[cls] = AutoWrappedModule
        enable_layer_vram_management(
            self.text_encoder.model,
            module_map=text_module_map,
            module_config=dict(
                offload_dtype=text_dtype,
                offload_device="cpu",
                onload_dtype=text_dtype,
                onload_device="cpu",
                computation_dtype=self.config.t5_dtype,
                computation_device=self.device,
                verbose=verbose,
            ),
        )

        dit_dtype = next(iter(self.model.parameters())).dtype
        dit_module_map = {
            nn.Linear: AutoWrappedLinear,
            nn.Conv3d: AutoWrappedModule,
            nn.LayerNorm: AutoWrappedModule,
        }
        for cls in self._collect_named_module_types(self.model, {"WanLayerNorm", "WanRMSNorm", "RMSNorm"}):
            dit_module_map[cls] = AutoWrappedModule
        enable_layer_vram_management(
            self.model,
            module_map=dit_module_map,
            module_config=dict(
                offload_dtype=dit_dtype,
                offload_device="cpu",
                onload_dtype=dit_dtype,
                onload_device=self.device,
                computation_dtype=self.param_dtype,
                computation_device=self.device,
                verbose=verbose,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dit_dtype,
                offload_device="cpu",
                onload_dtype=dit_dtype,
                onload_device="cpu",
                computation_dtype=self.param_dtype,
                computation_device=self.device,
                verbose=verbose,
            ),
        )

        vae_dtype = next(iter(self.vae.model.parameters())).dtype
        vae_module_map = {
            nn.Linear: AutoWrappedLinear,
            nn.Conv2d: AutoWrappedModule,
            nn.Conv3d: AutoWrappedModule,
            nn.SiLU: AutoWrappedModule,
            nn.Dropout: AutoWrappedModule,
        }
        for cls in self._collect_named_module_types(self.vae.model, {"RMS_norm", "RMSNorm", "Upsample", "CausalConv3d"}):
            vae_module_map[cls] = AutoWrappedModule
        enable_layer_vram_management(
            self.vae.model,
            module_map=vae_module_map,
            module_config=dict(
                offload_dtype=vae_dtype,
                offload_device="cpu",
                onload_dtype=vae_dtype,
                onload_device=self.device,
                computation_dtype=vae_dtype,
                computation_device=self.device,
                verbose=verbose,
            ),
        )

    def _register_vram_modules(self):
        if self.vram_manager is None:
            return
        self.vram_manager.register_module("text_encoder", self.text_encoder.model, pinned=self.t5_cpu)
        self.vram_manager.register_module("vae", self.vae.model)
        self.vram_manager.register_module("patch_embedding", self.model.patch_embedding)
        self.vram_manager.register_module("text_embedding", self.model.text_embedding)
        self.vram_manager.register_module("time_embedding", self.model.time_embedding)
        self.vram_manager.register_module("time_projection", self.model.time_projection)
        self.vram_manager.register_module("head", self.model.head)
        self.vram_manager.register_module("vace_patch_embedding", self.model.vace_patch_embedding)
        for idx, block in enumerate(self.model.blocks):
            self.vram_manager.register_module(f"dit_block_{idx}", block)
        for idx, block in enumerate(self.model.vace_blocks):
            self.vram_manager.register_module(f"vace_block_{idx}", block)

    def _load_runtime_modules(self, names, keep_resident=False):
        if self.vram_manager is None or not self.vram_manager.enabled:
            return
        self.vram_manager.load_modules(names)
        if not keep_resident:
            self.vram_manager.unload_modules(
                [name for name in self.vram_manager._modules.keys() if name not in set(names)],
                keep=names,
            )

    def _log_cuda_memory(self, tag):
        if not self.memory_log_enabled or not torch.cuda.is_available():
            return
        device = self.device
        allocated_mb = torch.cuda.memory_allocated(device) / 1024**2
        reserved_mb = torch.cuda.memory_reserved(device) / 1024**2
        max_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        max_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024**2
        logging.info(
            "[%s] %s | allocated=%.1fMB reserved=%.1fMB max_allocated=%.1fMB max_reserved=%.1fMB",
            self.memory_log_prefix,
            tag,
            allocated_mb,
            reserved_mb,
            max_allocated_mb,
            max_reserved_mb,
        )

    def vace_encode_frames(self, frames, ref_images, masks=None, vae=None):
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = vae.encode(frames)
        else:
            masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = vae.encode(inactive)
            reactive = vae.encode(reactive)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs)
                else:
                    ref_latent = vae.encode(refs)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None, vae_stride=None):
        vae_stride = self.vae_stride if vae_stride is None else vae_stride
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // vae_stride[0])
            height = 2 * (int(height) // (vae_stride[1] * 2))
            width = 2 * (int(width) // (vae_stride[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, vae_stride[1], width, vae_stride[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                vae_stride[1] * vae_stride[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def prepare_source(self, src_video, src_mask, src_ref_images, num_frames, image_size, device):
        area = image_size[0] * image_size[1]
        self.vid_proc.set_area(area)
        if area == 720*1280:
            self.vid_proc.set_seq_len(75600)
        elif area == 480*832:
            self.vid_proc.set_seq_len(32760)
        else:
            raise NotImplementedError(f'image_size {image_size} is not supported')

        image_size = (image_size[1], image_size[0])
        image_sizes = []
        for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
            if sub_src_mask is not None and sub_src_video is not None:
                src_video[i], src_mask[i], _, _, _ = self.vid_proc.load_video_pair(sub_src_video, sub_src_mask)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = src_mask[i].to(device)
                src_mask[i] = torch.clamp((src_mask[i][:1, :, :, :] + 1) / 2, min=0, max=1)
                image_sizes.append(src_video[i].shape[2:])
            elif sub_src_video is None:
                src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(image_size)
            else:
                src_video[i], _, _, _ = self.vid_proc.load_video(sub_src_video)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(src_video[i].shape[2:])

        for i, ref_images in enumerate(src_ref_images):
            if ref_images is not None:
                image_size = image_sizes[i]
                for j, ref_img in enumerate(ref_images):
                    if ref_img is not None:
                        ref_img = Image.open(ref_img).convert("RGB")
                        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
                        if ref_img.shape[-2:] != image_size:
                            canvas_height, canvas_width = image_size
                            ref_height, ref_width = ref_img.shape[-2:]
                            white_canvas = torch.ones((3, 1, canvas_height, canvas_width), device=device) # [-1, 1]
                            scale = min(canvas_height / ref_height, canvas_width / ref_width)
                            new_height = int(ref_height * scale)
                            new_width = int(ref_width * scale)
                            resized_image = F.interpolate(ref_img.squeeze(1).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0).unsqueeze(1)
                            top = (canvas_height - new_height) // 2
                            left = (canvas_width - new_width) // 2
                            white_canvas[:, :, top:top + new_height, left:left + new_width] = resized_image
                            ref_img = white_canvas
                        src_ref_images[i][j] = ref_img.to(device)
        return src_video, src_mask, src_ref_images

    def decode_latent(self, zs, ref_images=None, vae=None):
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(zs)
        else:
            assert len(zs) == len(ref_images)

        trimed_zs = []
        for z, refs in zip(zs, ref_images):
            if refs is not None:
                z = z[:, len(refs):, :, :]
            trimed_zs.append(z)

        return vae.decode(trimed_zs)



    def generate(self,
                 input_prompt,
                 input_frames,
                 input_masks,
                 input_ref_images,
                 size=(1280, 720),
                 frame_num=81,
                 context_scale=1.0,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        # F = frame_num
        # target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
        #                 size[1] // self.vae_stride[1],
        #                 size[0] // self.vae_stride[2])
        #
        # seq_len = math.ceil((target_shape[2] * target_shape[3]) /
        #                     (self.patch_size[1] * self.patch_size[2]) *
        #                     target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if self.vram_manager is not None and self.vram_manager.enabled:
            self.vram_manager.unload_modules()
        self._log_cuda_memory("generate:start")

        if not self.t5_cpu:
            if self.layer_vram_enabled:
                wrapped_model_onload(self.text_encoder.model)
            elif self.vram_manager is not None:
                self.vram_manager.load_modules(["text_encoder"])
            else:
                self.text_encoder.model.to(self.device)
            self._log_cuda_memory("text_encoder:before_encode")
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            self._log_cuda_memory("text_encoder:after_encode")
            if self.layer_vram_enabled:
                wrapped_model_offload(self.text_encoder.model)
            elif self.vram_manager is not None and self.vram_manager.enabled:
                self.vram_manager.unload_modules(["text_encoder"])
            elif offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # vace context encode
        if self.layer_vram_enabled:
            wrapped_model_onload(self.vae.model)
        elif self.vram_manager is not None and self.vram_manager.enabled:
            self.vram_manager.load_modules(["vae"])
        self._log_cuda_memory("vae:before_encode")
        z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks)
        m0 = self.vace_encode_masks(input_masks, input_ref_images)
        z = self.vace_latent(z0, m0)
        self._log_cuda_memory("vae:after_encode")
        if self.layer_vram_enabled:
            wrapped_model_offload(self.vae.model)
        elif self.vram_manager is not None and self.vram_manager.enabled:
            self.vram_manager.unload_modules(["vae"])

        target_shape = list(z0[0].shape)
        target_shape[0] = int(target_shape[0] / 2)
        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            if (not self.layer_vram_enabled) and (getattr(self.model, "_block_swap_enabled", False) or (self.vram_manager is not None and self.vram_manager.enabled)):
                self.model.prepare_runtime_modules(self.device)

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for step_idx, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)
                self._log_cuda_memory(f"denoise_step:{step_idx}:before_model")

                if self.layer_vram_enabled:
                    wrapped_model_onload(self.model)
                elif not getattr(self.model, "_block_swap_enabled", False) and not (self.vram_manager is not None and self.vram_manager.enabled):
                    self.model.to(self.device)
                lora_context = self.lora_manager.activated() if self.lora_manager is not None else noop_no_sync()
                with lora_context:
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale, **arg_null)[0]
                if self.layer_vram_enabled:
                    wrapped_model_offload(self.model)
                self._log_cuda_memory(f"denoise_step:{step_idx}:after_model")

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
                self._log_cuda_memory(f"denoise_step:{step_idx}:after_scheduler")

            x0 = latents
            if self.lora_manager is not None and self.lora_manager.active:
                self.lora_manager.remove()
            if self.layer_vram_enabled:
                wrapped_model_offload(self.model)
            elif self.vram_manager is not None and self.vram_manager.enabled:
                self.vram_manager.unload_modules()
            elif offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                if self.layer_vram_enabled:
                    wrapped_model_onload(self.vae.model)
                elif self.vram_manager is not None and self.vram_manager.enabled:
                    self.vram_manager.load_modules(["vae"])
                self._log_cuda_memory("vae:before_decode")
                videos = self.decode_latent(x0, input_ref_images)
                self._log_cuda_memory("vae:after_decode")
                if self.layer_vram_enabled:
                    wrapped_model_offload(self.vae.model)
                elif self.vram_manager is not None and self.vram_manager.enabled:
                    self.vram_manager.unload_modules(["vae"])

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        self._log_cuda_memory("generate:end")
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None


class WanVaceMP(WanVace):
    def __init__(
            self,
            config,
            checkpoint_dir,
            use_usp=False,
            ulysses_size=None,
            ring_size=None
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.use_usp = use_usp
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        self.in_q_list = None
        self.out_q = None
        self.inference_pids = None
        self.ulysses_size = ulysses_size
        self.ring_size = ring_size
        self.dynamic_load()

        self.device = 'cpu' if torch.cuda.is_available() else 'cpu'
        self.vid_proc = VaceVideoProcessor(
            downsample=tuple([x * y for x, y in zip(config.vae_stride, config.patch_size)]),
            min_area=720 * 1280,
            max_area=720 * 1280,
            min_fps=config.sample_fps,
            max_fps=config.sample_fps,
            zero_start=True,
            seq_len=75600,
            keep_last=True)


    def dynamic_load(self):
        if hasattr(self, 'inference_pids') and self.inference_pids is not None:
            return
        gpu_infer = os.environ.get('LOCAL_WORLD_SIZE') or torch.cuda.device_count()
        pmi_rank = int(os.environ['RANK'])
        pmi_world_size = int(os.environ['WORLD_SIZE'])
        in_q_list = [torch.multiprocessing.Manager().Queue() for _ in range(gpu_infer)]
        out_q = torch.multiprocessing.Manager().Queue()
        initialized_events = [torch.multiprocessing.Manager().Event() for _ in range(gpu_infer)]
        context = mp.spawn(self.mp_worker, nprocs=gpu_infer, args=(gpu_infer, pmi_rank, pmi_world_size, in_q_list, out_q, initialized_events, self), join=False)
        all_initialized = False
        while not all_initialized:
            all_initialized = all(event.is_set() for event in initialized_events)
            if not all_initialized:
                time.sleep(0.1)
        print('Inference model is initialized', flush=True)
        self.in_q_list = in_q_list
        self.out_q = out_q
        self.inference_pids = context.pids()
        self.initialized_events = initialized_events

    def transfer_data_to_cuda(self, data, device):
        if data is None:
            return None
        else:
            if isinstance(data, torch.Tensor):
                data = data.to(device)
            elif isinstance(data, list):
                data = [self.transfer_data_to_cuda(subdata, device) for subdata in data]
            elif isinstance(data, dict):
                data = {key: self.transfer_data_to_cuda(val, device) for key, val in data.items()}
        return data

    def mp_worker(self, gpu, gpu_infer, pmi_rank, pmi_world_size, in_q_list, out_q, initialized_events, work_env):
        try:
            world_size = pmi_world_size * gpu_infer
            rank = pmi_rank * gpu_infer + gpu
            print("world_size", world_size, "rank", rank, flush=True)

            torch.cuda.set_device(gpu)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=rank,
                world_size=world_size
            )

            from xfuser.core.distributed import (initialize_model_parallel,
                                                 init_distributed_environment)
            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())

            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=self.ring_size or 1,
                ulysses_degree=self.ulysses_size or 1
            )

            num_train_timesteps = self.config.num_train_timesteps
            param_dtype = self.config.param_dtype
            shard_fn = partial(shard_model, device_id=gpu)
            text_encoder = T5EncoderModel(
                text_len=self.config.text_len,
                dtype=self.config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=os.path.join(self.checkpoint_dir, self.config.t5_checkpoint),
                tokenizer_path=os.path.join(self.checkpoint_dir, self.config.t5_tokenizer),
                shard_fn=shard_fn if True else None)
            text_encoder.model.to(gpu)
            vae_stride = self.config.vae_stride
            patch_size = self.config.patch_size
            vae = WanVAE(
                vae_pth=os.path.join(self.checkpoint_dir, self.config.vae_checkpoint),
                device=gpu)
            logging.info(f"Creating VaceWanModel from {self.checkpoint_dir}")
            model = VaceWanModel.from_pretrained(self.checkpoint_dir)
            model.eval().requires_grad_(False)

            if self.use_usp:
                from xfuser.core.distributed import get_sequence_parallel_world_size
                from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                                usp_dit_forward,
                                                                usp_dit_forward_vace)
                for block in model.blocks:
                    block.self_attn.forward = types.MethodType(
                        usp_attn_forward, block.self_attn)
                for block in model.vace_blocks:
                    block.self_attn.forward = types.MethodType(
                        usp_attn_forward, block.self_attn)
                model.forward = types.MethodType(usp_dit_forward, model)
                model.forward_vace = types.MethodType(usp_dit_forward_vace, model)
                sp_size = get_sequence_parallel_world_size()
            else:
                sp_size = 1

            dist.barrier()
            model = shard_fn(model)
            sample_neg_prompt = self.config.sample_neg_prompt

            torch.cuda.empty_cache()
            event = initialized_events[gpu]
            in_q = in_q_list[gpu]
            event.set()

            while True:
                item = in_q.get()
                input_prompt, input_frames, input_masks, input_ref_images, size, frame_num, context_scale, \
                shift, sample_solver, sampling_steps, guide_scale, n_prompt, seed, offload_model = item
                input_frames = self.transfer_data_to_cuda(input_frames, gpu)
                input_masks = self.transfer_data_to_cuda(input_masks, gpu)
                input_ref_images = self.transfer_data_to_cuda(input_ref_images, gpu)

                if n_prompt == "":
                    n_prompt = sample_neg_prompt
                seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
                seed_g = torch.Generator(device=gpu)
                seed_g.manual_seed(seed)

                context = text_encoder([input_prompt], gpu)
                context_null = text_encoder([n_prompt], gpu)

                # vace context encode
                z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks, vae=vae)
                m0 = self.vace_encode_masks(input_masks, input_ref_images, vae_stride=vae_stride)
                z = self.vace_latent(z0, m0)

                target_shape = list(z0[0].shape)
                target_shape[0] = int(target_shape[0] / 2)
                noise = [
                    torch.randn(
                        target_shape[0],
                        target_shape[1],
                        target_shape[2],
                        target_shape[3],
                        dtype=torch.float32,
                        device=gpu,
                        generator=seed_g)
                ]
                seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                                    (patch_size[1] * patch_size[2]) *
                                    target_shape[1] / sp_size) * sp_size

                @contextmanager
                def noop_no_sync():
                    yield

                no_sync = getattr(model, 'no_sync', noop_no_sync)

                # evaluation mode
                with amp.autocast(dtype=param_dtype), torch.no_grad(), no_sync():

                    if sample_solver == 'unipc':
                        sample_scheduler = FlowUniPCMultistepScheduler(
                            num_train_timesteps=num_train_timesteps,
                            shift=1,
                            use_dynamic_shifting=False)
                        sample_scheduler.set_timesteps(
                            sampling_steps, device=gpu, shift=shift)
                        timesteps = sample_scheduler.timesteps
                    elif sample_solver == 'dpm++':
                        sample_scheduler = FlowDPMSolverMultistepScheduler(
                            num_train_timesteps=num_train_timesteps,
                            shift=1,
                            use_dynamic_shifting=False)
                        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                        timesteps, _ = retrieve_timesteps(
                            sample_scheduler,
                            device=gpu,
                            sigmas=sampling_sigmas)
                    else:
                        raise NotImplementedError("Unsupported solver.")

                    # sample videos
                    latents = noise

                    arg_c = {'context': context, 'seq_len': seq_len}
                    arg_null = {'context': context_null, 'seq_len': seq_len}

                    for _, t in enumerate(tqdm(timesteps)):
                        latent_model_input = latents
                        timestep = [t]

                        timestep = torch.stack(timestep)

                        model.to(gpu)
                        noise_pred_cond = model(
                            latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale, **arg_c)[
                            0]
                        noise_pred_uncond = model(
                            latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale,
                            **arg_null)[0]

                        noise_pred = noise_pred_uncond + guide_scale * (
                                noise_pred_cond - noise_pred_uncond)

                        temp_x0 = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                        latents = [temp_x0.squeeze(0)]

                    torch.cuda.empty_cache()
                    x0 = latents
                    if rank == 0:
                        videos = self.decode_latent(x0, input_ref_images, vae=vae)

                del noise, latents
                del sample_scheduler
                if offload_model:
                    gc.collect()
                    torch.cuda.synchronize()
                if dist.is_initialized():
                    dist.barrier()

                if rank == 0:
                    out_q.put(videos[0].cpu())

        except Exception as e:
            trace_info = traceback.format_exc()
            print(trace_info, flush=True)
            print(e, flush=True)



    def generate(self,
                 input_prompt,
                 input_frames,
                 input_masks,
                 input_ref_images,
                 size=(1280, 720),
                 frame_num=81,
                 context_scale=1.0,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):

        input_data = (input_prompt, input_frames, input_masks, input_ref_images, size, frame_num, context_scale,
                      shift, sample_solver, sampling_steps, guide_scale, n_prompt, seed, offload_model)
        for in_q in self.in_q_list:
            in_q.put(input_data)
        value_output = self.out_q.get()

        return value_output

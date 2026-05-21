# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from wan.modules.model import WanModel, WanAttentionBlock, sinusoidal_embedding_1d


class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        return c_skip, c
    
    
class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hint=None, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if hint is not None:
            x = x + hint * context_scale
        return x
    
    
class VaceWanModel(WanModel):
    @register_to_config
    def __init__(self,
                 vace_layers=None,
                 vace_in_dim=None,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        model_type = "t2v"   # TODO: Hard code for both preview and official versions.
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim,
                         num_heads, num_layers, window_size, qk_norm, cross_attn_norm, eps)

        self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # blocks
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                  self.cross_attn_norm, self.eps,
                                  block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None)
            for i in range(self.num_layers)
        ])

        # vace blocks
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                     self.cross_attn_norm, self.eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self._block_swap_enabled = False
        self._hint_cpu_offload = False
        self._runtime_device = None
        self._module_loader = None
        self._module_keep_resident = False
        self._dit_block_module_names = [f"dit_block_{i}" for i in range(self.num_layers)]
        self._vace_block_module_names = [f"vace_block_{i}" for i in range(len(self.vace_layers))]

    def enable_block_swap(self, enabled=True, device=None):
        self._block_swap_enabled = enabled
        self._runtime_device = device
        if enabled:
            for block in self.blocks:
                block.cpu()
            for block in self.vace_blocks:
                block.cpu()

    def enable_vace_hint_cpu_offload(self, enabled=True):
        self._hint_cpu_offload = enabled

    def set_module_loader(self, loader=None, keep_resident=False):
        self._module_loader = loader
        self._module_keep_resident = keep_resident

    def prepare_runtime_modules(self, device):
        self.patch_embedding.to(device)
        self.text_embedding.to(device)
        self.time_embedding.to(device)
        self.time_projection.to(device)
        self.head.to(device)
        self.vace_patch_embedding.to(device)

    def _load_runtime_module(self, names):
        if self._module_loader is not None:
            self._module_loader(names, keep_resident=self._module_keep_resident)

    def _run_runtime_block(self, block, *args, **kwargs):
        if not self._block_swap_enabled:
            return block(*args, **kwargs)

        if self._runtime_device is None:
            raise RuntimeError("Runtime device must be set when block swap is enabled.")

        block = block.to(self._runtime_device)
        output = block(*args, **kwargs)
        block.cpu()
        return output

    def forward_vace(
        self,
        x,
        vace_context,
        seq_len,
        kwargs
    ):
        # embeddings
        self._load_runtime_module(["vace_patch_embedding"])
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        hints = []
        for block_idx, block in enumerate(self.vace_blocks):
            self._load_runtime_module([self._vace_block_module_names[block_idx]])
            c_skip, c = self._run_runtime_block(block, c, **new_kwargs)
            if self._hint_cpu_offload:
                c_skip = c_skip.cpu()
            hints.append(c_skip)
        return hints

    def forward(
        self,
        x,
        t,
        vace_context,
        context,
        seq_len,
        vace_context_scale=1.0,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # if y is not None:
        #     x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        self._load_runtime_module(["patch_embedding", "text_embedding", "time_embedding", "time_projection"])
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # if clip_fea is not None:
        #     context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        #     context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        hints = self.forward_vace(x, vace_context, seq_len, kwargs)
        kwargs['context_scale'] = vace_context_scale

        for block_idx, block in enumerate(self.blocks):
            self._load_runtime_module([self._dit_block_module_names[block_idx]])
            hint = None
            if block.block_id is not None:
                hint = hints[block.block_id]
                if hint.device != x.device:
                    hint = hint.to(x.device, non_blocking=True)
            x = self._run_runtime_block(block, x, hint=hint, **kwargs)
            if block.block_id is not None and self._hint_cpu_offload:
                hints[block.block_id] = hint.cpu()

        # head
        self._load_runtime_module(["head"])
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

import os
import random
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import models
from ..utils import load_torch_checkpoint


def build_causal_mask(seq_length):
    mask = torch.empty(seq_length, seq_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)
    return mask


def init_weights(module, std=0.02):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            module.bias.data.zero_()
        if module.weight is not None:
            module.weight.data.fill_(1.0)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=1.0,
    filter_value=-float("inf"),
    min_tokens_to_keep=1,
):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    logits = logits / max(float(temperature), 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, attn_mask=None):
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.kv_cache:
            if self.k_cache is None and self.v_cache is None:
                k_cache = k
                v_cache = v
            else:
                k_cache = torch.cat([self.k_cache, k], dim=-2)
                v_cache = torch.cat([self.v_cache, v], dim=-2)

            self.k_cache = k_cache
            self.v_cache = v_cache
            k = k_cache
            v = v_cache

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch, seq_len, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm_final = norm_layer(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        return modulate(self.norm_final(x), shift, scale)


class RARBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x, attn_mask=None, c=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


@dataclass
class ModelArgs:
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    dropout: float = 0.1
    attn_drop: float = 0.1
    cond_dropout_prob: float = 0.1
    cond_dim: int = 1152
    max_cond_len: int = 729
    image_seq_len: int = 256
    vocab_size: int = 16384
    use_checkpoint: bool = False
    initializer_range: float = 0.02


class RARCondModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.image_seq_len = config.image_seq_len
        self.max_cond_len = config.max_cond_len
        self.vocab_size = config.vocab_size
        self.cls_token_num = config.max_cond_len
        self.random_ratio = 0.0

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        mlp_ratio = config.intermediate_size / config.hidden_size

        self.bos_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.uncond_embedding = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.condition_proj = nn.Sequential(
            nn.Linear(config.cond_dim, config.hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.condition_norm = norm_layer(config.hidden_size)

        self.target_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                RARBlock(
                    dim=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_norm=True,
                    proj_drop=config.dropout,
                    attn_drop=config.attn_drop,
                    norm_layer=norm_layer,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        total_len = 1 + config.max_cond_len + config.image_seq_len
        self.pos_embed = nn.Parameter(torch.zeros(1, total_len, config.hidden_size))
        self.target_aware_pos_embed = nn.Parameter(torch.zeros(1, total_len, config.hidden_size))
        self.timesteps_embeddings = nn.Parameter(torch.zeros(1, total_len, config.hidden_size))
        self.adaln_before_head = FinalLayer(config.hidden_size, norm_layer=norm_layer)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        self.apply(lambda module: init_weights(module, std=config.initializer_range))
        nn.init.trunc_normal_(self.pos_embed, mean=0.0, std=config.initializer_range)
        nn.init.trunc_normal_(self.target_aware_pos_embed, mean=0.0, std=config.initializer_range)
        nn.init.trunc_normal_(self.timesteps_embeddings, mean=0.0, std=config.initializer_range)
        nn.init.trunc_normal_(self.bos_token, mean=0.0, std=config.initializer_range)
        nn.init.trunc_normal_(self.uncond_embedding, mean=0.0, std=config.initializer_range)

        nn.init.constant_(self.adaln_before_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaln_before_head.adaLN_modulation[-1].bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        attn_mask = build_causal_mask(total_len)
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = True
            block.attn.reset_kv_cache()

    def disable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = False
            block.attn.reset_kv_cache()

    def set_random_ratio(self, new_ratio):
        self.random_ratio = float(new_ratio)

    def sample_orders(self, batch_size, seq_len, device):
        shuffled_orders = []
        for _ in range(batch_size):
            if random.random() < self.random_ratio:
                shuffled_orders.append(torch.randperm(seq_len, device=device))
            else:
                shuffled_orders.append(torch.arange(seq_len, device=device))
        return torch.stack(shuffled_orders, dim=0)

    def get_raster_orders(self, batch_size, seq_len, device):
        return torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)

    def shuffle(self, x, orders):
        batch_size, seq_len = x.shape[:2]
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, seq_len)
        return x[batch_indices, orders]

    def encode_condition(self, cond, force_uncond_mask=None):
        cond_tokens = self.condition_norm(self.condition_proj(cond.to(self.dtype)))
        batch_size, cond_len, hidden_size = cond_tokens.shape
        if force_uncond_mask is None:
            force_uncond_mask = torch.zeros(batch_size, dtype=torch.bool, device=cond_tokens.device)
        elif force_uncond_mask.dtype != torch.bool:
            force_uncond_mask = force_uncond_mask.to(dtype=torch.bool)

        if self.training and self.config.cond_dropout_prob > 0:
            dropout_mask = torch.rand(batch_size, device=cond_tokens.device) < self.config.cond_dropout_prob
            force_uncond_mask = force_uncond_mask | dropout_mask

        if force_uncond_mask.any():
            cond_tokens = cond_tokens.clone()
            cond_tokens[force_uncond_mask] = self.uncond_embedding.expand(force_uncond_mask.sum(), cond_len, hidden_size)

        cond_summary = cond_tokens.mean(dim=1)
        return cond_tokens, cond_summary

    def build_position_embeddings(self, batch_size, cond_len, target_len, orders=None):
        total_len = 1 + cond_len + target_len
        assert total_len <= self.pos_embed.shape[1], f"Requested sequence length {total_len} exceeds model capacity {self.pos_embed.shape[1]}"

        base_pos = self.pos_embed[:, :total_len].expand(batch_size, -1, -1)
        if target_len == 0:
            return base_pos

        prefix_len = 1 + cond_len
        target_pos = base_pos[:, prefix_len:]
        if orders is not None:
            target_pos = self.shuffle(target_pos, orders)
        pos_embed = torch.cat([base_pos[:, :prefix_len], target_pos], dim=1)

        target_aware = torch.zeros_like(pos_embed)
        target_aware_source = self.target_aware_pos_embed[:, prefix_len:prefix_len + target_len].expand(batch_size, -1, -1)
        if orders is not None:
            target_aware_source = self.shuffle(target_aware_source, orders)
        target_aware[:, prefix_len:] = target_aware_source

        return pos_embed + target_aware

    def build_condition_schedule(self, cond_summary, total_len):
        return cond_summary.unsqueeze(1) + self.timesteps_embeddings[:, :total_len]

    def forward_fn(self, idx, cond, return_labels=False, orders=None, is_sampling=False, force_uncond_mask=None):
        batch_size = cond.shape[0]
        cond_tokens, cond_summary = self.encode_condition(cond, force_uncond_mask=force_uncond_mask)
        cond_len = cond_tokens.shape[1]

        labels = None
        target_embeddings = cond_tokens.new_zeros(batch_size, 0, self.hidden_size)
        target_len = 0
        if idx is not None:
            target_len = idx.shape[1]
            target_embeddings = self.target_embeddings(idx)
            if orders is None:
                orders = self.get_raster_orders(batch_size, target_len, idx.device)
            labels = idx.clone()
            if not is_sampling:
                labels = self.shuffle(labels, orders)
                target_embeddings = self.shuffle(target_embeddings, orders)

        bos_tokens = self.bos_token.expand(batch_size, -1, -1)
        x = torch.cat([bos_tokens, cond_tokens, target_embeddings], dim=1)
        x = x + self.build_position_embeddings(batch_size, cond_len, target_len, orders=None if is_sampling else orders)

        total_len = x.shape[1]
        attn_mask = self.attn_mask[:total_len, :total_len]
        condition = self.build_condition_schedule(cond_summary, total_len)

        cache_ready = self.blocks[0].attn.kv_cache and self.blocks[0].attn.k_cache is not None
        if cache_ready:
            x = x[:, -1:]
            condition = condition[:, -1:]
            attn_mask = None

        for block in self.blocks:
            if self.config.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block.forward, x, attn_mask, condition, use_reentrant=False)
            else:
                x = block(x, attn_mask=attn_mask, c=condition)

        x = self.adaln_before_head(x, condition)
        logits = self.lm_head(x)

        if not is_sampling:
            logits = logits[:, cond_len:-1]

        if return_labels:
            return logits, labels
        return logits

    def forward(
        self,
        idx: Optional[torch.Tensor],
        cond_idx: Optional[torch.Tensor],
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        return_labels: bool = False,
    ):
        if cond_idx is None:
            raise ValueError("RARCondModel requires condition features.")

        if idx is None and not self.blocks[0].attn.kv_cache:
            raise ValueError("Training forward expects target token ids.")

        orders = None
        if idx is not None:
            orders = self.sample_orders(idx.shape[0], idx.shape[1], idx.device) if self.training else self.get_raster_orders(idx.shape[0], idx.shape[1], idx.device)

        outputs = self.forward_fn(
            idx,
            cond_idx,
            return_labels=return_labels,
            orders=orders,
            is_sampling=self.blocks[0].attn.kv_cache,
        )

        if return_labels:
            logits, labels = outputs
        else:
            logits, labels = outputs, None

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        if return_labels:
            return logits, labels
        return logits, loss

    @torch.no_grad()
    def sample(
        self,
        c,
        cfg_scale=4.0,
        cfg_interval=-1,
        temperature=1.02,
        top_k=0,
        top_p=1.0,
        seq_length=None,
        guidance_scale_pow=2.5,
    ):
        seq_length = self.image_seq_len if seq_length is None else seq_length
        if seq_length > self.image_seq_len:
            raise ValueError(f"Requested sequence length {seq_length} exceeds configured image_seq_len {self.image_seq_len}")

        device = c.device
        batch_size = c.shape[0]
        generated = torch.empty((batch_size, 0), dtype=torch.long, device=device)

        self.enable_kv_cache()
        try:
            for step in range(seq_length):
                if cfg_scale > 1.0:
                    step_cfg_scale = cfg_scale
                    if cfg_interval > -1 and step > cfg_interval:
                        step_cfg_scale = 1.0
                    scale_pow = torch.ones((1,), device=device) * guidance_scale_pow
                    scale_step = (1 - torch.cos(((step / max(seq_length, 1)) ** scale_pow) * torch.pi)) * 0.5
                    step_cfg_scale = (step_cfg_scale - 1.0) * scale_step + 1.0

                    force_uncond_mask = torch.cat(
                        [
                            torch.zeros(batch_size, dtype=torch.bool, device=device),
                            torch.ones(batch_size, dtype=torch.bool, device=device),
                        ],
                        dim=0,
                    )
                    logits = self.forward_fn(
                        generated.repeat(2, 1) if generated.numel() > 0 else None,
                        torch.cat([c, c], dim=0),
                        orders=None,
                        is_sampling=True,
                        force_uncond_mask=force_uncond_mask,
                    )
                    cond_logits, uncond_logits = logits[:batch_size], logits[batch_size:]
                    next_token_logits = uncond_logits[:, -1] + (cond_logits[:, -1] - uncond_logits[:, -1]) * step_cfg_scale
                else:
                    logits = self.forward_fn(generated if generated.numel() > 0 else None, c, orders=None, is_sampling=True)
                    next_token_logits = logits[:, -1]

                next_token = sample_logits(next_token_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                generated = torch.cat([generated, next_token], dim=1)
        finally:
            self.disable_kv_cache()

        return generated

    @classmethod
    def from_checkpoint(cls, ckpt, load_state_dict=True):
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f"checkpoint {ckpt} does not exist"
            ckpt = load_torch_checkpoint(ckpt, map_location="cpu")
        model = models.make(ckpt["model"], load_sd=load_state_dict)
        return model


def RAR_COND_B(**kwargs):
    return RARCondModel(ModelArgs(hidden_size=768, num_hidden_layers=24, num_attention_heads=16, intermediate_size=3072, **kwargs))


def RAR_COND_L(**kwargs):
    return RARCondModel(ModelArgs(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096, **kwargs))


def RAR_COND_XL(**kwargs):
    return RARCondModel(ModelArgs(hidden_size=1280, num_hidden_layers=32, num_attention_heads=16, intermediate_size=5120, **kwargs))


rar_models = {
    "rar-cond-B": RAR_COND_B,
    "rar-cond-L": RAR_COND_L,
    "rar-cond-XL": RAR_COND_XL,
}

models.models.update(rar_models)

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import cv2
import numpy as np

from ..utils import deprecate, logging
from ..utils.torch_utils import maybe_allow_in_graph
from .activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, SwiGLU
from .attention_processor import Attention, JointAttnProcessor2_0
from .embeddings import SinusoidalPositionalEmbedding
from .normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm

logger = logging.get_logger(__name__)


def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


@maybe_allow_in_graph
class GatedSelfAttentionDense(nn.Module):
    r"""
    A gated self-attention dense layer that combines visual features and object features.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        context_dim (`int`): The number of channels in the context.
        n_heads (`int`): The number of heads to use for attention.
        d_head (`int`): The number of channels in each head.
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        self.enabled = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        n_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.alpha_attn.tanh() * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :]
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        return x


@maybe_allow_in_graph
class JointTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, context_pre_only=False):
        super().__init__()

        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        self.norm1 = AdaLayerNormZero(dim)

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm"
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )
        if hasattr(F, "scaled_dot_product_attention"):
            processor = JointAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.double_self_attention = double_self_attention
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.only_cross_attention = only_cross_attention

        # We keep these boolean flags for backward-compatibility.
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )
        # if only_cross_attention is None:
        #     self.attn1.processor.set_query_mode('cache')

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif norm_type == "ada_norm_continuous":
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
                self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )  # is self-attn if encoder_hidden_states is none
        else:
            if norm_type == "ada_norm_single":  # For Latte
                self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
            else:
                self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )

        elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm"]:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prev_attention_masks: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        prev_encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        prev_encoder_attention_masks: Optional[torch.Tensor] = None,
        subject_token_idx: list = None,
        sigma: float = 0.0,
        multi_query_disentanglement: bool = False,
        process_overlap: bool = False,
        timestep: Optional[torch.LongTensor] = None,
        actual_timestep: int = None,
        block_idx: int = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        alpha : float = 1.0,
        object_gen: bool = False,
        utilize_cache: bool = False,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
        
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]
        # ADDED (For implementation of NoiseCollage style attention masking)
        # hidden_states_org = hidden_states.clone()
        # attention_mask = attention_mask[0,:].squeeze()at
        if attention_mask is not None:
            attention_mask_org = attention_mask.clone().to(torch.int)
        
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        # if attention_mask is not None:
        #     attn_output_org = norm_hidden_states.clone()
        #     non_zero_idx_attn_mask = attention_mask[0,:].squeeze().eq(0).nonzero().reshape(-1) # REVERSED
        #     query_masked = norm_hidden_states[:, non_zero_idx_attn_mask, :]
        #     zero_idx_attn_mask = attention_mask[0,:].squeeze().nonzero().reshape(-1)
        #     query_unmasked = norm_hidden_states[:, zero_idx_attn_mask, :]
        #     attn_output_masked = self.attn1(
        #         query_masked,
        #         encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        #         attention_mask=attention_mask,
        #         **cross_attention_kwargs,
        #     )

        #     attn_output_unmasked = self.attn1(
        #         query_unmasked,
        #         encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        #         attention_mask=attention_mask,
        #         **cross_attention_kwargs,
        #     )
        #     # attn_output = attn_output_masked + attn_output_unmasked
        #     attn_output_org[:, non_zero_idx_attn_mask,:] = attn_output_masked[:, non_zero_idx_attn_mask, :]
        #     attn_output_org[:, zero_idx_attn_mask, :] = attn_output_unmasked[:, zero_idx_attn_mask, :]
        #     attn_ouput = attn_output_org
        # else:
        if object_gen is True and utilize_cache is True:
            self.attn1.processor.set_query_mode('interpolate')
            self.attn1.processor.set_query_mode(mode="interpolate", vt=0.9 - actual_timestep * 0.04)
        elif object_gen is True and utilize_cache is False:
            self.attn1.processor.set_query_mode('cache')
        else:
            self.attn1.processor.set_query_mode('None')
        if object_gen is True and utilize_cache is True:
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=None,
                **cross_attention_kwargs,
            )
            
        else:
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=None, # originally attention_mask # FIXME,
                **cross_attention_kwargs,
            )

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            if attention_mask is not None: ### ADDED 
                if multi_query_disentanglement is True and prev_attention_masks is not None:
                    attn_output_org = norm_hidden_states.clone()
                    # Get zero_idx_attn_mask for the current iteration's query
                    non_zero_idx_attn_mask = attention_mask_org.nonzero().reshape(-1)
                    # Get all the other zero_idx_attn_mask for the previous iterations' queries
                    attn_mask_idx_for_each_layer = {}
                    for i in range(prev_attention_masks.shape[0] // attention_mask_org.shape[0]):
                        prev_attn_mask_iter = prev_attention_masks[i * attention_mask_org.shape[0] : (i + 1) * attention_mask_org.shape[0]]
                        non_zero_idx_attn_mask_iter = prev_attn_mask_iter.nonzero().reshape(-1)
                        attn_mask_idx_for_each_layer[i] = non_zero_idx_attn_mask_iter
                    
                    # Get subtracted mask for each iteration
                    real_non_zero_idx_attn_mask = {} # REVERSED ORDERING (CURRENT ITERATION FIRST)
                    real_non_zero_idx_overlap_attn_mask = {}
                    for i in range(int(prev_attention_masks.shape[0] // attention_mask_org.shape[0]) + 2):
                        if i == 0:
                            real_non_zero_idx_attn_mask[i] = non_zero_idx_attn_mask
                        elif i > 0 and i < int(prev_attention_masks.shape[0] // attention_mask_org.shape[0]) + 1:
                            real_non_zero_idx_attn_mask[i] = attn_mask_idx_for_each_layer[len(attn_mask_idx_for_each_layer) - i]
                            real_non_zero_idx_overlap_attn_mask[i] = []
                            for j in range(i):
                                tmp_mask = ~torch.isin(real_non_zero_idx_attn_mask[i], real_non_zero_idx_attn_mask[j])
                                real_non_zero_idx_attn_mask[i] = real_non_zero_idx_attn_mask[i][tmp_mask]
                                # Process overlapping regions
                                if process_overlap is True:
                                    real_non_zero_idx_overlap_attn_mask[i].append(real_non_zero_idx_attn_mask[i][torch.isin(real_non_zero_idx_attn_mask[i], real_non_zero_idx_attn_mask[j])])
                        else:
                            real_non_zero_idx_attn_mask[i] = torch.arange(0, attention_mask_org.shape[0], device=attention_mask_org.device)
                            # mask out the queries
                            for j in range(int(prev_attention_masks.shape[0] // attention_mask_org.shape[0]) + 1):
                                real_non_zero_idx_attn_mask[i] = real_non_zero_idx_attn_mask[i][~torch.isin(real_non_zero_idx_attn_mask[i], real_non_zero_idx_attn_mask[j])]
                    masked_query_set = {}
                    masked_overlapped_query_set = {}
                    for i in range(len(real_non_zero_idx_attn_mask)):
                        masked_query_set[i] = norm_hidden_states[:, real_non_zero_idx_attn_mask[i], :]
                    # process overlapping regions
                    if process_overlap is True:
                        for i in range(1, len(real_non_zero_idx_overlap_attn_mask)+1):
                            masked_overlapped_query_set = []
                            for j in range(len(real_non_zero_idx_overlap_attn_mask[i])):
                                masked_overlapped_query_set[i].append(norm_hidden_states[:, real_non_zero_idx_overlap_attn_mask[i][j], :])
                    # Get the masked output for each iteration
                    all_encoder_hidden_states = {}
                    overlapped_encoder_hidden_states = {}
                    for i in range(int(prev_encoder_hidden_states.shape[2] // encoder_hidden_states.shape[2]) + 1):
                        if i == 0:
                            all_encoder_hidden_states[i] = encoder_hidden_states
                        else:
                            prev_len = int(prev_encoder_hidden_states.shape[2] // encoder_hidden_states.shape[2])
                            all_encoder_hidden_states[i] = prev_encoder_hidden_states[:,:,(prev_len - i) * 1152 : (prev_len - i + 1) * 1152]
                            # Process overlapped region's encoder hidden states
                            if process_overlap is True:
                                overlapped_encoder_hidden_states[i] = []
                                for j in range(i):
                                    if j == 0:
                                        overlapped_encoder_hidden_states[i].append((prev_encoder_hidden_states[:,:,(prev_len - i) * 1152 : (prev_len - i + 1) * 1152], encoder_hidden_states))
                                    else:
                                        overlapped_encoder_hidden_states[i].append((prev_encoder_hidden_states[:,:,(prev_len - i) * 1152 : (prev_len - i + 1) * 1152], prev_encoder_hidden_states[:,:,(prev_len - j) * 1152 : (prev_len - j + 1) * 1152])) # 
                    all_encoder_attention_masks = {}
                    overlapped_encoder_attention_masks = {}
                    for i in range(int(prev_encoder_hidden_states.shape[2] // encoder_hidden_states.shape[2]) + 1):
                        if i == 0:
                            all_encoder_attention_masks[i] = encoder_attention_mask
                        else:
                            prev_len = int(prev_encoder_hidden_states.shape[2] // encoder_hidden_states.shape[2])
                            all_encoder_attention_masks[i] = prev_encoder_attention_masks[:,(prev_len - i):(prev_len - i + 1)]
                            # Process overlapped region's encoder attention mask
                            if process_overlap is True:
                                overlapped_encoder_attention_masks[i] = []
                                for j in range(i):
                                    if j == 0:
                                        overlapped_encoder_attention_masks[i].append((prev_encoder_attention_masks[:,(prev_len - i):(prev_len - i + 1)], encoder_attention_mask))
                                    else:
                                        overlapped_encoder_attention_masks[i].append((prev_encoder_attention_masks[:,(prev_len - i):(prev_len - i + 1)], prev_encoder_attention_masks[:,(prev_len - j):(prev_len - j + 1)]))
                                    
                    masked_output_set = {}
                    for i in range(len(real_non_zero_idx_attn_mask)):
                        masked_output_set[i] = self.attn2(
                            masked_query_set[i],
                            encoder_hidden_states=all_encoder_hidden_states[i],
                            attention_mask=all_encoder_attention_masks[i],
                            **cross_attention_kwargs,
                        )
                    if process_overlap is True:
                        overlapped_mask_output_set = {}
                        cnt = 0
                        for i in range(1, len(real_non_zero_idx_overlap_attn_mask)+1):
                            overlapped_mask_output_set = []
                            for j in range(len(real_non_zero_idx_overlap_attn_mask[i])):
                                # Process overlapped encoder hidden states
                                enc_state_below, enc_state_above = overlapped_encoder_hidden_states[i][j]
                                # Process overlapped encoder attention masks
                                enc_attn_mask_below, enc_attn_mask_above = overlapped_encoder_attention_masks[i][j]
                                # Merge the encoder hidden states and masks
                                enc_merged = (enc_state_below + enc_state_above) / 2
                                enc_attn_mask_merged = (enc_attn_mask_below + enc_attn_mask_above) / 2
                                # Overlapped cross attention
                                overlapped_mask_output_set[i].append(self.attn2(
                                    masked_overlapped_query_set[i][j],
                                    encoder_hidden_states=enc_merged,
                                    attention_mask=enc_attn_mask_merged,
                                    **cross_attention_kwargs,
                                ))

                    # get them summed up
                    for i in range(len(real_non_zero_idx_attn_mask)):
                        attn_output_org[:, real_non_zero_idx_attn_mask[i], :] = masked_output_set[i]
                    # overlapped part
                    if process_overlap is True:
                        for i in range(len(real_non_zero_idx_overlap_attn_mask)):
                            for j in range(len(real_non_zero_idx_overlap_attn_mask[i])):
                                attn_output_org[:, real_non_zero_idx_overlap_attn_mask[i][j], :] = overlapped_mask_output_set[i][j]
                    attn_output = attn_output_org
                else: # cattn_masking == True
                    attn_output_org = norm_hidden_states.clone()
                    non_zero_idx_attn_mask = attention_mask_org.nonzero().reshape(-1) # REVERSED -> Not reversed now # FIXME
                    query_masked = norm_hidden_states[:, non_zero_idx_attn_mask, :] * alpha
                    
                    zero_idx_attn_mask = attention_mask_org.eq(0.).nonzero().reshape(-1)
                    query_unmasked = norm_hidden_states[:, zero_idx_attn_mask, :] * (1 / alpha)
                    # Update query_mask (through cross-attention map based attention guided mask)
                    # if actual_timestep > 10 and block_idx >= 14 :
                    #     attn_output_zeros = torch.zeros_like(norm_hidden_states)
                    #     attn_output_zeros[:, non_zero_idx_attn_mask, :] = query_masked
                    #     # np.save('attention_map_induce/image_step_{}_{}.npy'.format(actual_timestep, block_idx), attn_output_zeros.cpu().detach().numpy())
                    #     # np.save('attention_map_induce/text_step_{}_{}.npy'.format(actual_timestep, block_idx), encoder_hidden_states.cpu().detach().numpy())
                    #     attn_output_zeros = attn_output_zeros.reshape(2, 64, 64, -1) # FIXME
                    #     # attn_output_zeros = attn_output_zeros.reshape(2, 32, 32, -1)
                    #     image = attn_output_zeros[1,...] # use only the classifier guided part
                    #     text = encoder_hidden_states[1,...] # use only the classifier guided part
                    #     # get the subject token list
                    #     if subject_token_idx is None: # FIXME : should be updated on __call__
                    #         # average the tokens
                    #         text = text.mean(dim=0)
                    #     elif len(subject_token_idx) == 1:
                    #         text = text[subject_token_idx[0],...]
                    #     else:
                    #         text = text[subject_token_idx,...].mean(dim=0)
                    #     mid_rep = torch.einsum('ijk,k->ij', image, text)
                    #     # Calculate statistical distribution
                    #     mid_rep = (mid_rep - mid_rep.mean()) / (mid_rep.max() - mid_rep.min()) # Normalizing
                    #     mid_rep_mean = float(mid_rep.mean())
                    #     mid_rep_std = float(mid_rep.std())
                    #     # thresholding
                    #     mid_rep_mask = ((mid_rep - mid_rep_mean) / mid_rep_std > sigma).float() # FIXME update sigma on __init__
                    #     mid_rep_nonzeros = mid_rep_mask.reshape(-1).nonzero().reshape(-1)
                    #     # get the non-zero index (considering non_zero_idx_attn_mask)
                    #     # UPDATE ALL THE MASK AGAIN
                    #     non_zero_idx_attn_mask = mid_rep_nonzeros[torch.isin(mid_rep_nonzeros, non_zero_idx_attn_mask)]
                    #     # # optional dilation
                    #     # mask_before_dilation = torch.zeros_like(mid_rep_mask).reshape(-1)
                    #     # mask_before_dilation[non_zero_idx_attn_mask] = 1
                    #     # mask_before_dilation_2d = mask_before_dilation.reshape(64, 64)
                    #     # kernel = np.ones((5, 5), np.uint8)
                    #     # mask_after_dilation_2d = torch.from_numpy(cv2.dilate(mask_before_dilation_2d.cpu().detach().numpy(), kernel, iterations=1)).to(mask_before_dilation.device)
                    #     # mask_after_dilation = mask_after_dilation_2d.reshape(-1).nonzero().reshape(-1)
                        
                    #     # non_zero_idx_attn_mask = mask_after_dilation[torch.isin(mask_after_dilation, non_zero_idx_attn_mask)]
                    #     query_masked = norm_hidden_states[:, non_zero_idx_attn_mask, :]
                    #     all_idx_mask = torch.ones((4096,), device=non_zero_idx_attn_mask.device).reshape(-1).nonzero().reshape(-1)
                    #     # FIXME
                    #     # all_idx_mask = torch.ones((1024,), device=non_zero_idx_attn_mask.device).reshape(-1).nonzero().reshape(-1)
                    #     zero_idx_attn_mask = all_idx_mask[~torch.isin(all_idx_mask, non_zero_idx_attn_mask)]
                    #     query_unmasked = norm_hidden_states[:, zero_idx_attn_mask, :]
                    

                    attn_output_masked = self.attn2(
                        query_masked,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask,
                        **cross_attention_kwargs,
                    )
                    
                    attn_output_unmasked = self.attn2(
                        query_unmasked,
                        encoder_hidden_states=prev_encoder_hidden_states,
                        attention_mask=prev_encoder_attention_masks,
                        **cross_attention_kwargs,
                    )
                    attn_output_org[:, non_zero_idx_attn_mask,:] = attn_output_masked
                    attn_output_org[:, zero_idx_attn_mask, :] = attn_output_unmasked
                    attn_output = attn_output_org
                
            else: # attention mask is None
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class LuminaFeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        hidden_size (`int`):
            The dimensionality of the hidden layers in the model. This parameter determines the width of the model's
            hidden representations.
        intermediate_size (`int`): The intermediate dimension of the feedforward layer.
        multiple_of (`int`, *optional*): Value to ensure hidden dimension is a multiple
            of this value.
        ffn_dim_multiplier (float, *optional*): Custom multiplier for hidden
            dimension. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        multiple_of: Optional[int] = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        inner_dim = int(2 * inner_dim / 3)
        # custom hidden_size factor multiplier
        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        self.linear_1 = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )
        self.linear_2 = nn.Linear(
            inner_dim,
            dim,
            bias=False,
        )
        self.linear_3 = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )
        self.silu = FP32SiLU()

    def forward(self, x):
        return self.linear_2(self.silu(self.linear_1(x)) * self.linear_3(x))


@maybe_allow_in_graph
class TemporalBasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block for video like data.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        time_mix_inner_dim (`int`): The number of channels for temporal attention.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
    """

    def __init__(
        self,
        dim: int,
        time_mix_inner_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.is_res = dim == time_mix_inner_dim

        self.norm_in = nn.LayerNorm(dim)

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.ff_in = FeedForward(
            dim,
            dim_out=time_mix_inner_dim,
            activation_fn="geglu",
        )

        self.norm1 = nn.LayerNorm(time_mix_inner_dim)
        self.attn1 = Attention(
            query_dim=time_mix_inner_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = nn.LayerNorm(time_mix_inner_dim)
            self.attn2 = Attention(
                query_dim=time_mix_inner_dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(time_mix_inner_dim)
        self.ff = FeedForward(time_mix_inner_dim, activation_fn="geglu")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        # chunk dim should be hardcoded to 1 to have better speed vs. memory trade-off
        self._chunk_dim = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        batch_frames, seq_length, channels = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * seq_length, num_frames, channels)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)

        if self._chunk_size is not None:
            hidden_states = _chunked_feed_forward(self.ff_in, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None, :].reshape(batch_size, seq_length, num_frames, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * num_frames, seq_length, channels)

        return hidden_states


class SkipFFTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        kv_input_dim: int,
        kv_input_dim_proj_use_bias: bool,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        if kv_input_dim != dim:
            self.kv_mapper = nn.Linear(kv_input_dim, dim, kv_input_dim_proj_use_bias)
        else:
            self.kv_mapper = None

        self.norm1 = RMSNorm(dim, 1e-06)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            out_bias=attention_out_bias,
        )

        self.norm2 = RMSNorm(dim, 1e-06)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

    def forward(self, hidden_states, encoder_hidden_states, cross_attention_kwargs):
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

        if self.kv_mapper is not None:
            encoder_hidden_states = self.kv_mapper(F.silu(encoder_hidden_states))

        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **cross_attention_kwargs,
        )

        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **cross_attention_kwargs,
        )

        hidden_states = attn_output + hidden_states

        return hidden_states


@maybe_allow_in_graph
class FreeNoiseTransformerBlock(nn.Module):
    r"""
    A FreeNoise Transformer block.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        cross_attention_dim (`int`, *optional*):
            The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to be used in feed-forward.
        num_embeds_ada_norm (`int`, *optional*):
            The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (`bool`, defaults to `False`):
            Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, defaults to `False`):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, defaults to `False`):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, defaults to `False`):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
        ff_inner_dim (`int`, *optional*):
            Hidden dimension of feed-forward MLP.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in feed-forward MLP.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in attention output project layer.
        context_length (`int`, defaults to `16`):
            The maximum number of frames that the FreeNoise block processes at once.
        context_stride (`int`, defaults to `4`):
            The number of frames to be skipped before starting to process a new batch of `context_length` frames.
        weighting_scheme (`str`, defaults to `"pyramid"`):
            The weighting scheme to use for weighting averaging of processed latent frames. As described in the
            Equation 9. of the [FreeNoise](https://arxiv.org/abs/2310.15169) paper, "pyramid" is the default setting
            used.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        context_length: int = 16,
        context_stride: int = 4,
        weighting_scheme: str = "pyramid",
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.double_self_attention = double_self_attention
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.only_cross_attention = only_cross_attention

        self.set_free_noise_properties(context_length, context_stride, weighting_scheme)

        # We keep these boolean flags for backward-compatibility.
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )  # is self-attn if encoder_hidden_states is none

        # 3. Feed-forward
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def _get_frame_indices(self, num_frames: int) -> List[Tuple[int, int]]:
        frame_indices = []
        for i in range(0, num_frames - self.context_length + 1, self.context_stride):
            window_start = i
            window_end = min(num_frames, i + self.context_length)
            frame_indices.append((window_start, window_end))
        return frame_indices

    def _get_frame_weights(self, num_frames: int, weighting_scheme: str = "pyramid") -> List[float]:
        if weighting_scheme == "flat":
            weights = [1.0] * num_frames

        elif weighting_scheme == "pyramid":
            if num_frames % 2 == 0:
                # num_frames = 4 => [1, 2, 2, 1]
                mid = num_frames // 2
                weights = list(range(1, mid + 1))
                weights = weights + weights[::-1]
            else:
                # num_frames = 5 => [1, 2, 3, 2, 1]
                mid = (num_frames + 1) // 2
                weights = list(range(1, mid))
                weights = weights + [mid] + weights[::-1]

        elif weighting_scheme == "delayed_reverse_sawtooth":
            if num_frames % 2 == 0:
                # num_frames = 4 => [0.01, 2, 2, 1]
                mid = num_frames // 2
                weights = [0.01] * (mid - 1) + [mid]
                weights = weights + list(range(mid, 0, -1))
            else:
                # num_frames = 5 => [0.01, 0.01, 3, 2, 1]
                mid = (num_frames + 1) // 2
                weights = [0.01] * mid
                weights = weights + list(range(mid, 0, -1))
        else:
            raise ValueError(f"Unsupported value for weighting_scheme={weighting_scheme}")

        return weights

    def set_free_noise_properties(
        self, context_length: int, context_stride: int, weighting_scheme: str = "pyramid"
    ) -> None:
        self.context_length = context_length
        self.context_stride = context_stride
        self.weighting_scheme = weighting_scheme

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0) -> None:
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

        # hidden_states: [B x H x W, F, C]
        device = hidden_states.device
        dtype = hidden_states.dtype

        num_frames = hidden_states.size(1)
        frame_indices = self._get_frame_indices(num_frames)
        frame_weights = self._get_frame_weights(self.context_length, self.weighting_scheme)
        frame_weights = torch.tensor(frame_weights, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)
        is_last_frame_batch_complete = frame_indices[-1][1] == num_frames

        # Handle out-of-bounds case if num_frames isn't perfectly divisible by context_length
        # For example, num_frames=25, context_length=16, context_stride=4, then we expect the ranges:
        #    [(0, 16), (4, 20), (8, 24), (10, 26)]
        if not is_last_frame_batch_complete:
            if num_frames < self.context_length:
                raise ValueError(f"Expected {num_frames=} to be greater or equal than {self.context_length=}")
            last_frame_batch_length = num_frames - frame_indices[-1][1]
            frame_indices.append((num_frames - self.context_length, num_frames))

        num_times_accumulated = torch.zeros((1, num_frames, 1), device=device)
        accumulated_values = torch.zeros_like(hidden_states)

        for i, (frame_start, frame_end) in enumerate(frame_indices):
            # The reason for slicing here is to ensure that if (frame_end - frame_start) is to handle
            # cases like frame_indices=[(0, 16), (16, 20)], if the user provided a video with 19 frames, or
            # essentially a non-multiple of `context_length`.
            weights = torch.ones_like(num_times_accumulated[:, frame_start:frame_end])
            weights *= frame_weights

            hidden_states_chunk = hidden_states[:, frame_start:frame_end]

            # Notice that normalization is always applied before the real computation in the following blocks.
            # 1. Self-Attention
            norm_hidden_states = self.norm1(hidden_states_chunk)

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            hidden_states_chunk = attn_output + hidden_states_chunk
            if hidden_states_chunk.ndim == 4:
                hidden_states_chunk = hidden_states_chunk.squeeze(1)

            # 2. Cross-Attention
            if self.attn2 is not None:
                norm_hidden_states = self.norm2(hidden_states_chunk)

                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states_chunk = attn_output + hidden_states_chunk

            if i == len(frame_indices) - 1 and not is_last_frame_batch_complete:
                accumulated_values[:, -last_frame_batch_length:] += (
                    hidden_states_chunk[:, -last_frame_batch_length:] * weights[:, -last_frame_batch_length:]
                )
                num_times_accumulated[:, -last_frame_batch_length:] += weights[:, -last_frame_batch_length]
            else:
                accumulated_values[:, frame_start:frame_end] += hidden_states_chunk * weights
                num_times_accumulated[:, frame_start:frame_end] += weights

        # TODO(aryan): Maybe this could be done in a better way.
        #
        # Previously, this was:
        # hidden_states = torch.where(
        #    num_times_accumulated > 0, accumulated_values / num_times_accumulated, accumulated_values
        # )
        #
        # The reasoning for the change here is `torch.where` became a bottleneck at some point when golfing memory
        # spikes. It is particularly noticeable when the number of frames is high. My understanding is that this comes
        # from tensors being copied - which is why we resort to spliting and concatenating here. I've not particularly
        # looked into this deeply because other memory optimizations led to more pronounced reductions.
        hidden_states = torch.cat(
            [
                torch.where(num_times_split > 0, accumulated_split / num_times_split, accumulated_split)
                for accumulated_split, num_times_split in zip(
                    accumulated_values.split(self.context_length, dim=1),
                    num_times_accumulated.split(self.context_length, dim=1),
                )
            ],
            dim=1,
        ).to(dtype)

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

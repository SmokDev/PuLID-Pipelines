# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttnProcessor(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        id_embedding=None,
        id_scale=1.0,
        pulid_ortho: str = None,
        pulid_num_zero: int = 16,
        pulid_mode:str = None,
        avalible_pulid = True,
        **kwargs
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class PuLIDAttnProcessor(torch.nn.Module):
    r"""
    Attention processor for ID-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
    """

    def __init__(self, original_attn_processor, hidden_size, cross_attention_dim=None):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.original_attn_processor = original_attn_processor

        self.id_to_k = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.id_to_v = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.hidden_size = hidden_size

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        id_embedding=None,
        id_scale=1.0,
        pulid_ortho: str = None,
        pulid_num_zero: int = 16,
        pulid_mode:str = None,
        avalible_pulid: bool = True,
        **kwargs
    ):
        original_hidden_states = self.original_attn_processor(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            **kwargs
        )

        # for id embedding
        if id_embedding is not None and id_scale > 0 and avalible_pulid:
            if pulid_mode is not None:
                if pulid_mode == 'fidelity':
                    pulid_num_zero = 8
                    pulid_ortho = 'v2'
                elif pulid_mode == 'extremely style':
                    pulid_num_zero = 16
                    pulid_ortho = 'v1'
                else:
                    raise ValueError("Unsupported pulid mode. Use 'fidelity' or 'extremely style'.")


            residual = hidden_states

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, _ , _ = (hidden_states.shape)
            query = attn.to_q(hidden_states)

            head_dim = self.hidden_size // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        
            if pulid_num_zero == 0:
                id_key = self.id_to_k(id_embedding).to(query.dtype)
                id_value = self.id_to_v(id_embedding).to(query.dtype)
            else:
                zero_tensor = torch.zeros(
                    (id_embedding.size(0), pulid_num_zero, id_embedding.size(-1)),
                    dtype=id_embedding.dtype,
                    device=id_embedding.device,
                )
                id_key = self.id_to_k(torch.cat((id_embedding, zero_tensor), dim=1)).to(query.dtype)
                id_value = self.id_to_v(torch.cat((id_embedding, zero_tensor), dim=1)).to(query.dtype)

            id_key = id_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            id_value = id_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            id_hidden_states = F.scaled_dot_product_attention(
                query, id_key, id_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            id_hidden_states = id_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            id_hidden_states = id_hidden_states.to(query.dtype)

            if pulid_ortho == 'v1':
                hidden_states = hidden_states + id_scale * id_hidden_states
            elif pulid_ortho == 'v2':
                orig_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                id_hidden_states = id_hidden_states.to(torch.float32)
                attn_map = query @ id_key.transpose(-2, -1)
                attn_mean = attn_map.softmax(dim=-1).mean(dim=1)
                attn_mean = attn_mean[:, :, :5].sum(dim=-1, keepdim=True)
                projection = (
                    torch.sum((hidden_states * id_hidden_states), dim=-2, keepdim=True)
                    / torch.sum((hidden_states * hidden_states), dim=-2, keepdim=True)
                    * hidden_states
                )
                orthogonal = id_hidden_states + (attn_mean - 1) * projection
                hidden_states = hidden_states + id_scale * orthogonal
                hidden_states = hidden_states.to(orig_dtype)
            else:
                orig_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                id_hidden_states = id_hidden_states.to(torch.float32)
                projection = (
                    torch.sum((hidden_states * id_hidden_states), dim=-2, keepdim=True)
                    / torch.sum((hidden_states * hidden_states), dim=-2, keepdim=True)
                    * hidden_states
                )
                orthogonal = id_hidden_states - projection
                hidden_states = hidden_states + id_scale * orthogonal
                hidden_states = hidden_states.to(orig_dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return original_hidden_states + hidden_states
        else:
            return original_hidden_states


        

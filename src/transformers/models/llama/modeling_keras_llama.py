# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" Keras 3.0 LLaMA model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import keras

from ...activations_keras import ACT2FN
from ...utils import (
    logging
)
# from ...keras_utils import ALL_LAYERNORM_LAYERS
from .configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


class KerasLlamaRMSNorm(keras.layers.Layer):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.add_variable(
            shape=hidden_size,
            initializer="ones",
            trainable=True,
            name="weight",
        )            
        self.variance_epsilon = eps

    def call(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = keras.ops.cast(hidden_states, "float32")
        variance = keras.ops.mean(
            keras.ops.power(hidden_states, 2),
            axis=-1,
            keepdims=True,
        )
        hidden_states = hidden_states * keras.ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * keras.ops.cast(hidden_states, input_dtype)


# ALL_LAYERNORM_LAYERS.append(KerasLlamaRMSNorm)


class KerasLlamaRotaryEmbedding(keras.layers.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # TODO (ariG23498): use add_weight or constant to handle inv_freq
        self.inv_freq = 1.0 / (
            self.base ** (keras.ops.arange(start=0, stop=self.dim, step=2, dtype="float32") / dim)
        )

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=keras.backend.floatx(),
        )
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = keras.ops.arange(
            self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = keras.ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = keras.ops.concatenate((freqs, freqs), axis=-1)
        
        self.cos_cached = self.add_weight(
            shape=keras.ops.shape(emb),
            initializer="ones",
            trainable=False,
            dtype=dtype,
            name="cos_cached",
        )
        self.cos_cached.assign(keras.ops.cos(emb))
        
        self.sin_cached = self.add_weight(
            shape=keras.ops.shape(emb),
            initializer="ones",
            trainable=False,
            dtype=dtype,
            name="sin_cached",
        )
        self.sin_cached.assign(keras.ops.sin(emb))

    def call(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            keras.ops.cast(self.cos_cached[:seq_len], dtype=x.dtype),
            keras.ops.cast(self.sin_cached[:seq_len], dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return keras.ops.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = keras.ops.expand_dims(cos[position_ids], axis=unsqueeze_dim)
    sin = keras.ops.expand_dims(sin[position_ids], axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed    


class KerasLlamaMLP(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = keras.layers.Dense(self.intermediate_size, bias=False, name="gate_proj")
        self.up_proj = keras.layers.Dense(self.intermediate_size, bias=False, name="up_proj")
        self.down_proj = keras.layers.Dense(self.hidden_size, bias=False, name="down_proj")
        self.act_fn = ACT2FN[config.hidden_act]

    def call(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = keras.ops.split(self.gate_proj.weight, indices_or_sections=slice, axis=0)
            up_proj_slices = keras.ops.split(self.up_proj.weight, indices_or_sections=slice, axis=0)
            down_proj_slices = keras.ops.split(self.down_proj.weight, indices_or_sections=slice, axis=1)

            gate_proj = keras.ops.concatenate(
                [keras.ops.matmul(x, keras.ops.transpose(gate_proj_slices[i])) for i in range(self.config.pretraining_tp)], axis=-1
            )
            up_proj = keras.ops.cat(
                [keras.ops.matmul(x, keras.ops.transpose(up_proj_slices[i])) for i in range(self.config.pretraining_tp)],
                axis=-1
            )

            intermediate_states = keras.ops.split((self.act_fn(gate_proj) * up_proj), indices_or_sections=slice, axis=2)
            down_proj = [
                keras.ops.matmul(intermediate_states[i], keras.ops.transpose(down_proj_slices[i])) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: keras.KerasTensor, n_rep: int) -> keras.KerasTensor:
    batch, num_key_value_heads, slen, head_dim = keras.ops.shape(hidden_states)
    if n_rep == 1:
        return hidden_states
    hidden_states = keras.ops.broadcast_to(hidden_states[:, :, None, :, :], shape=[batch, num_key_value_heads, n_rep, slen, head_dim])
    return keras.ops.reshape(hidden_states, new_shape=[batch, num_key_value_heads * n_rep, slen, head_dim])


class KerasLlamaAttention(keras.layers.Layer):
    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = keras.layers.Dense(self.num_heads * self.head_dim, bias=config.attention_bias, name="q_proj")
        self.k_proj = keras.layers.Dense(self.num_key_value_heads * self.head_dim, bias=config.attention_bias, name="k_proj")
        self.v_proj = keras.layers.Dense(self.num_key_value_heads * self.head_dim, bias=config.attention_bias, name="v_proj")
        self.o_proj = keras.layers.Dense(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias, name="o_proj")
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = KerasLlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            raise ValueError(f"Aritra did not support RoPE scaling yet!")

    def _shape(self, tensor: keras.KerasTensor, seq_len: int, bsz: int):
        return keras.ops.transpose(
            keras.ops.reshape(tensor, new_shape=[bsz, seq_len, self.num_heads, self.head_dim]),
            axes=[1, 2]
        )

    def call(
        self,
        hidden_states: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        past_key_value: Optional[Tuple[keras.KerasTensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[keras.KerasTensor, Optional[keras.KerasTensor], Optional[Tuple[keras.KerasTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = keras.ops.shape(hidden_states)

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = keras.ops.split(self.q_proj.weight, 
                indices_or_sections=(self.num_heads * self.head_dim) // self.config.pretraining_tp,
                axis=0,
            )
            key_slices = keras.ops.split(self.k_proj.weight, indices_or_sections=key_value_slicing, axis=0)
            value_slices = keras.ops.split(self.v_proj.weight, indices_or_sections=key_value_slicing, axis=0)

            query_states = [keras.ops.matmul(hidden_states, keras.ops.transpose(query_slices[i])) for i in range(self.config.pretraining_tp)]
            query_states = keras.ops.concatenate(query_states, axis=-1)

            key_states = [keras.ops.matmul(hidden_states, keras.ops.transpose(key_slices[i])) for i in range(self.config.pretraining_tp)]
            key_states = keras.ops.concatenate(key_states, axis=-1)

            value_states = [keras.ops.matmul(hidden_states, keras.ops.transpose(value_slices[i])) for i in range(self.config.pretraining_tp)]
            value_states = keras.ops.concatenate(value_states, axis=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = keras.ops.transpose(
            keras.ops.reshape(query_states, new_shape=[bsz, q_len, self.num_heads, self.head_dim]),
            axes=[1, 2],
        )
        key_states = keras.ops.transpose(
            keras.ops.reshape(key_states, new_shape=[bsz, q_len, self.num_key_value_heads, self.head_dim]),
            axes=[1, 2],
        )
        value_states = keras.ops.transpose(
            keras.ops.reshape(value_states, new_shape=[bsz, q_len, self.num_key_value_heads, self.head_dim]),
            axes=[1, 2],
        )

        kv_seq_len = keras.ops.shape(key_states)[-2]
        if past_key_value is not None:
            kv_seq_len += keras.ops.shape(past_key_value[0])[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = keras.ops.concatenate([past_key_value[0], key_states], axis=2)
            value_states = keras.ops.concatenate([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = keras.ops.matmul(query_states, keras.ops.transpose(key_states, axes=[2, 3])) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = keras.ops.cast(
            keras.ops.softmax(attn_weights, dim=-1, dtype="float32"),
            dtype=query_states.dtype
        )
        attn_weights = keras.layers.Dropout(rate=self.attention_dropout)(attn_weights)
        attn_output = keras.ops.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = keras.ops.transpose(attn_output, axes=[1, 2])

        attn_output = keras.ops.reshape(attn_output, new_shape=[bsz, q_len, self.hidden_size])

        if self.config.pretraining_tp > 1:
            attn_output = keras.ops.split(attn_output, indices_or_sections=self.hidden_size // self.config.pretraining_tp, axis=2)
            o_proj_slices = keras.ops.split(self.o_proj.weight, indices_or_sections=self.hidden_size // self.config.pretraining_tp, axis=1)
            attn_output = sum(
                [keras.ops.matmul(attn_output[i], keras.ops.transpose(o_proj_slices[i])) for i in range(self.config.pretraining_tp)]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class KerasLlamaDecoderLayer(keras.layers.Layer):
    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = KerasLlamaAttention(config, name="self_attn")
        self.mlp = KerasLlamaMLP(config, name="mlp")
        self.input_layer_norm = KerasLlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            name="input_layer_norm"
        )
        self.post_attention_layernorm = KerasLlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            name="post_attention_layernorm"
        )

    def call(
        self,
        hidden_states: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        past_key_value: Optional[Tuple[keras.KerasTensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
        )-> Tuple[keras.KerasTensor, Optional[Tuple[keras.KerasTensor, keras.KerasTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead."
            )
        
        residual = hidden_states

        hidden_states = self.input_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

# TODO (ariG23498): adding KerasPreTrainedModel
class KerasLlamaPreTrainedModel(KerasPreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "llama"

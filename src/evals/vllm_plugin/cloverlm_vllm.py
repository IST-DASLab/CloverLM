
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


def _build_rope_cos_sin(
    positions: torch.Tensor,
    d_head: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    js = torch.arange(d_head // 2, device=device, dtype=torch.float32)
    theta = 1.0 / (1024.0 ** (2.0 * js / d_head))
    phi = positions.float().unsqueeze(-1) * theta.unsqueeze(0)
    cos = torch.cos(phi).repeat_interleave(2, dim=-1)
    sin = torch.sin(phi).repeat_interleave(2, dim=-1)
    return cos, sin


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x_rot = torch.empty_like(x)
    x_rot[..., 0::2] = -x[..., 1::2]
    x_rot[..., 1::2] = x[..., 0::2]
    return (x * cos + x_rot * sin).to(x.dtype)



class CloverLMAttention(nn.Module):

    def __init__(
        self,
        d: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        self.num_heads = num_heads // tp_size
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim

        total_q_size = num_heads * head_dim
        total_kv_size = num_kv_heads * head_dim

        if num_kv_heads % tp_size == 0:
            self.num_kv_heads = num_kv_heads // tp_size
            kv_linear_cls = ColumnParallelLinear
        else:
            self.num_kv_heads = num_kv_heads
            kv_linear_cls = ReplicatedLinear

        self.kv_size = self.num_kv_heads * head_dim

        self.lq = ColumnParallelLinear(
            d, total_q_size, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.lq",
        )
        self.lk = kv_linear_cls(
            d, total_kv_size, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.lk",
        )
        self.lv = kv_linear_cls(
            d, total_kv_size, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.lv",
        )
        self.lo = RowParallelLinear(
            total_q_size, d, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.lo",
        )

        self.scale = nn.Parameter(
            torch.empty(1, self.num_heads, 1, 1),
            requires_grad=False,
        )
        heads_per_tp = self.num_heads

        def _scale_weight_loader(param, loaded_weight):
            start = tp_rank * heads_per_tp
            end = start + heads_per_tp
            param.data.copy_(loaded_weight[:, start:end, :, :])

        self.scale.weight_loader = _scale_weight_loader

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=head_dim,
            scale=1.0,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q, _ = self.lq(hidden_states)
        k, _ = self.lk(hidden_states)
        v, _ = self.lv(hidden_states)

        cos, sin = _build_rope_cos_sin(
            positions, self.head_dim, hidden_states.device,
        )

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)

        q = _apply_rope(q, cos.unsqueeze(1), sin.unsqueeze(1))
        k = _apply_rope(k, cos.unsqueeze(1), sin.unsqueeze(1))

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # scale: (1, heads, 1, 1) → broadcast over (tokens, heads, head_dim)
        q = q * self.scale.squeeze(-1)

        q = q.reshape(-1, self.q_size)
        k = k.reshape(-1, self.kv_size)

        attn_output = self.attn(q, k, v)
        output, _ = self.lo(attn_output)
        return output


class CloverLMMLP(nn.Module):

    def __init__(
        self,
        d: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        d_hidden = 4 * d
        self.l1 = ColumnParallelLinear(
            d, d_hidden, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.l1.0",
        )
        self.l2 = RowParallelLinear(
            d_hidden, d, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.l2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.l1(x)
        x = F.relu(x) ** 2
        x, _ = self.l2(x)
        return x


class CloverLMBlock(nn.Module):

    def __init__(
        self,
        d: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.mhsa = CloverLMAttention(
            d, num_heads, num_kv_heads, head_dim,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mhsa",
        )
        self.out_att_norm = RMSNorm(d)
        self.mlp = CloverLMMLP(
            d,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.out_mlp_norm = RMSNorm(d)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Post-norm attention residual
        attn_out = self.mhsa(positions, hidden_states)
        attn_out = self.out_att_norm(attn_out)
        hidden_states = hidden_states + attn_out

        # Post-norm MLP residual
        mlp_out = self.mlp(hidden_states)
        mlp_out = self.out_mlp_norm(mlp_out)
        hidden_states = hidden_states + mlp_out

        return hidden_states


class CloverLMModel(nn.Module):

    def __init__(
        self,
        config,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        d = config.heads * config.d_head

        self.emb = VocabParallelEmbedding(
            config.vocab_size, d,
            quant_config=quant_config,
            prefix=f"{prefix}.emb",
        )
        self.blocks = nn.ModuleList([
            CloverLMBlock(
                d, config.heads,
                config.heads // config.ratio,
                config.d_head,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{i}",
            )
            for i in range(config.num_blocks)
        ])
        self.out_norm = RMSNorm(d)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.emb(input_ids)

        for block in self.blocks:
            hidden_states = block(positions, hidden_states)

        hidden_states = self.out_norm(hidden_states)
        return hidden_states



_HF_TO_VLLM = WeightsMapper(
    orig_to_new_prefix={"transformer.": "model."},
)


class CloverLMForCausalLM_vLLM(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        d = config.heads * config.d_head
        self.config = config

        self.model = CloverLMModel(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}model",
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size, d, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}lm_head",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        if getattr(config, "weight_tying", True):
            self.lm_head.weight = self.model.emb.weight

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()

        skip_prefixes = set()
        if getattr(self.config, "weight_tying", True):
            skip_prefixes.add("transformer.linear.weight")

        skipped = []
        unmapped = []
        for hf_name, loaded_weight in weights:
            if hf_name in skip_prefixes:
                skipped.append(hf_name)
                continue

            # Map HuggingFace names → vLLM names
            vllm_name = hf_name.replace("transformer.", "model.", 1)

            # In HuggingFace model, MLP l1 is Sequential(Linear, ReLU²),
            # so the linear weight is at "mlp.l1.0.weight".  In our vLLM
            # model l1 is a flat ColumnParallelLinear → "mlp.l1.weight".
            vllm_name = vllm_name.replace(".mlp.l1.0.", ".mlp.l1.")

            if vllm_name not in params_dict:
                unmapped.append(f"{hf_name} -> {vllm_name}")
                continue

            param = params_dict[vllm_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(vllm_name)

        not_loaded = set(params_dict.keys()) - loaded
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Loaded %d/%d params, skipped %d, unmapped %d, "
                     "not_loaded %d",
                     len(loaded), len(params_dict), len(skipped),
                     len(unmapped), len(not_loaded))
        if unmapped:
            logger.warning("Unmapped HF keys: %s", unmapped)
        if not_loaded:
            logger.warning("Params not loaded: %s", sorted(not_loaded))

        return loaded

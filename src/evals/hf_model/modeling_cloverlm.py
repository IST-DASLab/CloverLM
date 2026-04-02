
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_cloverlm import CloverLMConfig
from .fake_quartet import FakeQuartetLinear


# ── NVFP4 dequantization for checkpoint loading ─────────────────────────────

def _dequant_nvfp4_state_dict(raw_sd, dtype=torch.bfloat16):
    """Dequantize NVFP4-packed tensors using quartet2's _dq_fp4 on GPU.

    The micro-scales are stored in cuBLAS blocked layout; quartet2's _dq_fp4
    handles the unblocking correctly.
    """
    from quartet2.linear import _dq_fp4

    scale2_bases = {k.removesuffix("_scale_2") for k in raw_sd if k.endswith("_scale_2")}
    result = {}
    for key, tensor in raw_sd.items():
        if key.endswith(("_scale", "_scale_2")):
            continue
        if key in scale2_bases:
            fp4 = tensor.cuda()
            scales = raw_sd[f"{key}_scale"].cuda()
            ts = raw_sd[f"{key}_scale_2"].float().item()
            result[key] = _dq_fp4(fp4, scales, ts).to(dtype).cpu()
        else:
            result[key] = tensor.to(dtype) if tensor.is_floating_point() else tensor
    return result



def _sphere_norm(X, dim=-1):
    return F.normalize(X, dim=dim)


class _ReLU2(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


def _make_linear(in_f, out_f, bias, quartet_2_impl):
    if quartet_2_impl == "pseudoquant":
        return FakeQuartetLinear(in_f, out_f, bias)
    elif quartet_2_impl == "quartet2":
        try:
            from quartet2.linear import Quartet_II_linear
        except ImportError as e:
            e.add_note("Quartet_II_linear import failed. Install the latest quartet2 from https://github.com/IST-DASLab/Quartet-II")
            raise e
        
        return Quartet_II_linear(in_f, out_f, bias)
    elif quartet_2_impl in ("bf16", None, ""):
        return nn.Linear(in_f, out_f, bias=bias)
    else:
        raise ValueError(f"Unsupported quartet_2_impl: {quartet_2_impl}")


def _build_rope(context, d_head, device):
    ms = torch.arange(context, device=device, dtype=torch.float32)
    js = torch.arange(d_head // 2, device=device, dtype=torch.float32)
    theta = 1.0 / (1024.0 ** (2.0 * js / d_head))
    phi = ms[:, None] @ theta[None, :]
    cos = torch.cos(phi).repeat_interleave(2, dim=1)
    sin = torch.sin(phi).repeat_interleave(2, dim=1)
    return torch.stack((cos, sin))


def _apply_rope(X, rope):
    X_ = torch.empty_like(X)
    X_[..., 0::2] = -X[..., 1::2]
    X_[..., 1::2] = X[..., 0::2]
    return (X * rope[0] + X_ * rope[1]).to(X.dtype)



class _MLP(nn.Module):

    def __init__(self, d, d_hidden, quartet_2_impl):
        super().__init__()
        self.l1 = nn.Sequential(_make_linear(d, d_hidden, False, quartet_2_impl), _ReLU2())
        self.l2 = _make_linear(d_hidden, d, False, quartet_2_impl)

    def forward(self, x):
        return self.l2(self.l1(x))



class MHSA(nn.Module):
    def __init__(self, heads, d_head, ratio, quartet_2_impl):
        super().__init__()
        self.heads = heads
        self.d_head = d_head
        self.d = heads * d_head
        self.groups = heads // ratio
        d_kv = self.groups * d_head

        self.lq = _make_linear(self.d, self.d, False, quartet_2_impl)
        self.lk = _make_linear(self.d, d_kv, False, quartet_2_impl)
        self.lv = _make_linear(self.d, d_kv, False, quartet_2_impl)
        self.lo = _make_linear(self.d, self.d, False, quartet_2_impl)

        self.scale = nn.Parameter(torch.full((1, heads, 1, 1), sqrt(d_head)))

    def forward(self, X, rope, attn_backend):
        B = X.shape[0] if X.dim() == 3 else 1
        ctx = X.shape[-2]

        Q = self.lq(X).unflatten(-1, (self.heads, self.d_head)).movedim(-3, -2)
        K = self.lk(X).unflatten(-1, (self.groups, self.d_head)).movedim(-3, -2)
        V = self.lv(X).unflatten(-1, (self.groups, self.d_head)).movedim(-3, -2)

        Q = _apply_rope(Q, rope)
        K = _apply_rope(K, rope)
        Q = _sphere_norm(Q)
        K = _sphere_norm(K)

        Q_shape = Q.shape
        Q = self.scale * Q
        Q = Q.reshape(Q_shape)

        if attn_backend == "pytorch":
            K = K.repeat_interleave(self.heads // self.groups, dim=-3)
            V = V.repeat_interleave(self.heads // self.groups, dim=-3)
            Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=1.0)
            Y = Y.movedim(-3, -2).flatten(-2, -1)
        elif attn_backend in ("flash2", "flash3", "flash4"):
            Q = Q.movedim(-3, -2).reshape(-1, ctx, self.heads, self.d_head)
            K = K.movedim(-3, -2).reshape(-1, ctx, self.groups, self.d_head)
            V = V.movedim(-3, -2).reshape(-1, ctx, self.groups, self.d_head)

            dtype = Q.dtype if Q.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
            if attn_backend == "flash2":
                try:
                    import flash_attn
                except ImportError as e:
                    e.add_note(f"Can't run `attn_backend=flash2` because can't import flash_attn")
                    raise e
                Y = flash_attn.flash_attn_func(Q.to(dtype), K.to(dtype), V.to(dtype), causal=True, softmax_scale=1.0)
            elif attn_backend == "flash3":
                import importlib
                try:
                    _fa3 = importlib.import_module("flash_attn_interface")
                except ImportError as e:
                    e.add_note(f"Can't run `attn_backend=flash3` because can't import flash_attn_interface")
                    raise e
                Y = _fa3.flash_attn_func(Q.to(dtype), K.to(dtype), V.to(dtype), causal=True, softmax_scale=1.0)
            elif attn_backend == "flash4":
                import importlib
                try:
                    _fa4 = importlib.import_module("flash_attn.cute")
                except ImportError as e:
                    e.add_note(f"Can't run `attn_backend=flash4` because can't import flash_attn.cute")
                    raise e
                Y = _fa4.flash_attn_func(Q.to(dtype), K.to(dtype), V.to(dtype), causal=True, softmax_scale=1.0)[0]
            Y = Y.to(Q.dtype).flatten(-2, -1)

        return self.lo(Y)



class _Block(nn.Module):

    def __init__(self, heads, d_head, ratio, quartet_2_impl):
        super().__init__()
        d = heads * d_head

        self.mhsa = MHSA(heads, d_head, ratio, quartet_2_impl)
        self.out_att_norm = nn.RMSNorm(d, elementwise_affine=True)

        self.mlp = _MLP(d, 4 * d, quartet_2_impl)
        self.out_mlp_norm = nn.RMSNorm(d, elementwise_affine=True)

    def forward(self, X, rope, attn_backend):
        Y = self.out_att_norm(self.mhsa(X, rope, attn_backend))
        Y = X + Y
        Z = self.out_mlp_norm(self.mlp(Y))
        return Y + Z



class _Transformer(nn.Module):

    def __init__(self, vocab_size, num_blocks, heads, d_head, ratio,
                 max_context, std, quartet_2_impl, weight_tying, attn_backend):
        super().__init__()
        self.d_head = d_head
        self.attn_backend = attn_backend
        d = heads * d_head

        self.emb = nn.Embedding(vocab_size, d)
        self.blocks = nn.Sequential(*[
            _Block(heads, d_head, ratio, quartet_2_impl) for _ in range(num_blocks)
        ])
        self.out_norm = nn.RMSNorm(d, elementwise_affine=True)
        self.linear = nn.Linear(d, vocab_size, bias=False)

        if weight_tying:
            self.emb.weight = self.linear.weight

        for name, p in self.named_parameters():
            parent_name, _, suffix = name.rpartition(".")
            parent = self.get_submodule(parent_name)
            if isinstance(parent, (nn.Linear, nn.Embedding)) and suffix == "weight":
                nn.init.normal_(p, 0, std)
            elif isinstance(parent, nn.RMSNorm) and suffix == "weight":
                nn.init.ones_(p)
            elif p.ndim == 4:
                nn.init.constant_(p, sqrt(d_head))

        if quartet_2_impl:
            for m in self.modules():
                if isinstance(m, (nn.LayerNorm, nn.RMSNorm, nn.Embedding)):
                    m.to(torch.bfloat16)

    def forward(self, ids):
        ctx = ids.shape[-1]
        rope = _build_rope(ctx, self.d_head, device=ids.device)

        X = self.emb(ids)
        for block in self.blocks:
            X = block(X, rope, self.attn_backend)
        X = self.out_norm(X)
        return self.linear(X)



class CloverLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = CloverLMConfig
    supports_gradient_checkpointing = False
    _no_split_modules = ["_Block"]
    _tied_weights_keys = {"transformer.linear.weight": "transformer.emb.weight"}
    _tp_plan = {}

    def __init__(self, config: CloverLMConfig):
        super().__init__(config)
        self.transformer = _Transformer(
            vocab_size=config.vocab_size,
            num_blocks=config.num_blocks,
            heads=config.heads,
            d_head=config.d_head,
            ratio=config.ratio,
            max_context=config.max_context,
            std=0.02,
            quartet_2_impl=config.quartet_2_impl,
            weight_tying=config.weight_tying,
            attn_backend=config.attn_backend,
        )
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        import os
        from safetensors import safe_open

        st_path = os.path.join(str(pretrained_model_name_or_path), "model.safetensors")
        if not os.path.exists(st_path):
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        with safe_open(st_path, framework="pt") as f:
            if not any(k.endswith("_scale_2") for k in f.keys()):
                return super().from_pretrained(
                    pretrained_model_name_or_path, *args, **kwargs,
                )

        from safetensors.torch import load_file

        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True,
            )

        # Apply config overrides from kwargs (e.g. attn_backend, quartet_2_impl)
        for key in list(kwargs.keys()):
            if hasattr(config, key):
                setattr(config, key, kwargs.pop(key))
        kwargs.pop("trust_remote_code", None)

        target_dtype = kwargs.pop("torch_dtype", None)
        if target_dtype is None:
            target_dtype = torch.bfloat16
        if isinstance(target_dtype, str):
            target_dtype = getattr(torch, target_dtype)

        device_map = kwargs.pop("device_map", None)

        raw = load_file(st_path)
        state_dict = _dequant_nvfp4_state_dict(raw, target_dtype)

        model = cls(config)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(target_dtype)

        if device_map is not None:
            if isinstance(device_map, str) and device_map != "auto":
                model = model.to(device_map)
            elif isinstance(device_map, dict):
                device = next(iter(device_map.values()))
                model = model.to(device)
            elif device_map == "auto":
                from accelerate import dispatch_model, infer_auto_device_map
                device_map_computed = infer_auto_device_map(model)
                model = dispatch_model(model, device_map=device_map_computed)

        model.eval()
        return model

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits = self.transformer(input_ids)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def _supports_default_dynamic_cache(self):
        return False

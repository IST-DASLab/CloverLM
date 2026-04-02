

import argparse
import json
import os
import re
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROJECT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean expected, got '{v}'")


# ── NVFP4 quantization ──────────────────────────────────────────────────────

NVFP4_GROUP_SIZE = 16


def _quantize_weight_nvfp4(w):
    """Quantize a 2-D weight tensor to NVFP4 on GPU via quartet2.

    Returns (fp4, micro_scales, tensor_scale) moved back to CPU.
      fp4          : float4_e2m1fn_x2  [rows, cols // 2]
      micro_scales : float8_e4m3fn     [rows, cols // 16]
      tensor_scale : float32           scalar
    """
    from quartet2.quant import quant_fp4, NVFP4QuantMode

    assert w.ndim == 2
    rows, cols = w.shape
    pad_r, pad_c = (-rows) % 128, (-cols) % 128
    if pad_r or pad_c:
        w = F.pad(w, (0, pad_c, 0, pad_r))

    w_gpu = w.to(device="cuda", dtype=torch.bfloat16).contiguous()
    nvfp4 = quant_fp4(w_gpu, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)

    fp4 = nvfp4.fp4[:rows, :cols // 2].view(torch.float4_e2m1fn_x2)
    scales = nvfp4.micro_scales[:rows, :cols // NVFP4_GROUP_SIZE]
    return fp4.cpu(), scales.cpu(), nvfp4.tensor_scale.cpu()


def _is_quantizable(key):
    """True for 2-D block weights (not embedding, output head, or norms)."""
    if "emb.weight" in key or key.endswith("linear.weight"):
        return False
    if ".weight" in key:
        return True
    return False


def detect_architecture(state_dict):
    """Infer architecture params from checkpoint keys/shapes."""
    vocab_size, d = state_dict["emb.weight"].shape
    heads = state_dict["blocks.0.mhsa.scale"].shape[1]
    d_head = d // heads
    zeta = d_head // 8

    d_kv_per_group = state_dict["blocks.0.mhsa.lk.weight"].shape[0]
    groups = d_kv_per_group // d_head
    ratio = heads // groups

    max_block = max(
        int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")
    )
    num_blocks = max_block + 1

    return {
        "vocab_size": vocab_size,
        "num_blocks": num_blocks,
        "heads": heads,
        "d_head": d_head,
        "zeta": zeta,
        "ratio": ratio,
        "d": d,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert .pt checkpoint to HuggingFace format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("checkpoint", type=os.path.abspath,
                    help="Path to .pt checkpoint")
    p.add_argument("output_dir", type=os.path.abspath,
                    help="Output HuggingFace model directory")
    p.add_argument("--vocab_size", type=int, default=None,
                    help="Override auto-detected vocab_size")
    p.add_argument("--num_blocks", type=int, default=None,
                    help="Override auto-detected num_blocks")
    p.add_argument("--heads", type=int, default=None,
                    help="Override auto-detected heads")
    p.add_argument("--ratio", type=int, default=None,
                    help="Override auto-detected ratio")
    p.add_argument("--zeta", type=int, default=None,
                    help="Override auto-detected zeta")
    p.add_argument("--scale_type", default="1/sqrt(d)")
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--quartet_2_impl", type=str, default="pseudoquant")
    p.add_argument("--attn_backend", default="pytorch",
                    choices=["pytorch", "flash2", "flash3", "flash4"])
    p.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "bfloat16", "float16"],
                    help="Dtype for non-quantized weights (default: bfloat16)")
    p.add_argument("--nvfp4", action="store_true",
                    help="Quantize block weights to NVFP4 (4-bit with "
                         "two-level micro-block scaling for Blackwell)")
    return p.parse_args()


def _patch_imports(src_text, filename):
    if filename == "exp_transformer.py":
        src_text = src_text.replace(
            "from . import mlp",
            "from . import exp_mlp as mlp",
        )
        src_text = re.sub(
            r"import fake_quartet as fq",
            "from . import fake_quartet as fq",
            src_text,
        )
        # Remove quartet2 imports — replace the whole if-block's quartet
        # branch with a pass (we never set quartet=True in HF mode)
        src_text = re.sub(
            r"import quartet2\.linear",
            "pass  # quartet2 not available in HF mode",
            src_text,
        )
    elif filename == "exp_mlp.py":
        src_text = re.sub(
            r"import fake_quartet as fq",
            "from . import fake_quartet as fq",
            src_text,
        )
        src_text = re.sub(
            r"import quartet2\.linear",
            "pass  # quartet2 not available in HF mode",
            src_text,
        )
    return src_text


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1.  Load the raw checkpoint
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, weights_only=True,
                            map_location="cpu")

    # ------------------------------------------------------------------
    # 1b. Auto-detect architecture (CLI args override if provided)
    # ------------------------------------------------------------------
    arch = detect_architecture(state_dict)
    vocab_size  = args.vocab_size  if args.vocab_size  is not None else arch["vocab_size"]
    num_blocks  = args.num_blocks  if args.num_blocks  is not None else arch["num_blocks"]
    heads       = args.heads       if args.heads       is not None else arch["heads"]
    ratio       = args.ratio       if args.ratio       is not None else arch["ratio"]
    zeta        = args.zeta        if args.zeta        is not None else arch["zeta"]
    d_head      = zeta * 8

    n_params = sum(v.numel() for v in state_dict.values())
    print(f"  auto-detected: blocks={arch['num_blocks']}  heads={arch['heads']}  "
          f"zeta={arch['zeta']}  ratio={arch['ratio']}  "
          f"d={arch['d']}  ~{n_params/1e9:.2f}B params")

    target_dtype = getattr(torch, args.dtype)

    # Prefix every key with "transformer." so it matches the HF wrapper.
    # Clone shared tensors (weight tying: emb.weight == linear.weight)
    # so safetensors doesn't complain about shared memory.
    hf_state_dict = {}

    if args.nvfp4:
        print(f"\n  NVFP4 quantization enabled (group_size={NVFP4_GROUP_SIZE})")
        quantized_keys = []
        for k, v in state_dict.items():
            hf_key = f"transformer.{k}"
            if v.ndim == 2 and _is_quantizable(hf_key):
                packed, scale, scale2 = _quantize_weight_nvfp4(v)
                hf_state_dict[hf_key] = packed
                hf_state_dict[f"{hf_key}_scale"] = scale
                hf_state_dict[f"{hf_key}_scale_2"] = scale2
                quantized_keys.append(hf_key)
            else:
                hf_state_dict[hf_key] = v.clone().to(target_dtype)
        print(f"  quantized {len(quantized_keys)} weight tensors")
    else:
        for k, v in state_dict.items():
            hf_state_dict[f"transformer.{k}"] = v.clone().to(target_dtype)

    size_gb = sum(v.numel() * v.element_size() for v in hf_state_dict.values()) / 1e9
    print(f"  output size: {size_gb:.2f} GB")

    # ------------------------------------------------------------------
    # 2.  Save weights as safetensors
    # ------------------------------------------------------------------
    safetensors_path = os.path.join(args.output_dir, "model.safetensors")
    print(f"Saving weights → {safetensors_path}")
    save_file(hf_state_dict, safetensors_path)

    # ------------------------------------------------------------------
    # 3.  Write config.json (via the HF config class)
    # ------------------------------------------------------------------
    sys.path.insert(0, SCRIPT_DIR)
    from configuration_cloverlm import CloverLMConfig

    config = CloverLMConfig(
        vocab_size=vocab_size,
        num_blocks=num_blocks,
        heads=heads,
        d_head=d_head,
        ratio=ratio,
        scale_type=args.scale_type,
        max_context=args.context,
        quartet_2_impl=args.quartet_2_impl,
        weight_tying=True,
        attn_backend=args.attn_backend,
        torch_dtype=args.dtype,
        architectures=["CloverLMForCausalLM"],
        auto_map={
            "AutoConfig": "configuration_cloverlm.CloverLMConfig",
            "AutoModelForCausalLM":
                "modeling_cloverlm.CloverLMForCausalLM",
            "AutoTokenizer": [
                "tokenization_cloverlm.CloverLMTokenizer", None],
        },
    )
    config.save_pretrained(args.output_dir)
    print(f"Saved config.json → {args.output_dir}/config.json")

    if args.nvfp4:
        hf_quant_config = {
            "producer": {"name": "cloverlm_converter", "version": "1.0"},
            "quantization": {
                "quant_algo": "NVFP4",
                "kv_cache_quant_algo": None,
                "group_size": NVFP4_GROUP_SIZE,
                "exclude_modules": ["emb", "linear"],
            },
        }
        hf_qc_path = os.path.join(args.output_dir, "hf_quant_config.json")
        with open(hf_qc_path, "w") as f:
            json.dump(hf_quant_config, f, indent=2)
        print(f"Saved hf_quant_config.json → {hf_qc_path}")

    # ------------------------------------------------------------------
    # 3b. Write tokenizer_config.json
    # ------------------------------------------------------------------
    tok_config = {
        "tokenizer_class": "CloverLMTokenizer",
        "auto_map": {
            "AutoTokenizer": [
                "tokenization_cloverlm.CloverLMTokenizer", None]
        },
        "use_fast": False,
    }
    tok_config_path = os.path.join(args.output_dir, "tokenizer_config.json")
    with open(tok_config_path, "w") as f:
        json.dump(tok_config, f, indent=2)
    print(f"Saved tokenizer_config.json → {tok_config_path}")

    # ------------------------------------------------------------------
    # 4.  Bundle Python source files
    # ------------------------------------------------------------------
    # Files that go straight into the model dir
    files_to_copy = {
        # dst filename                   src path
        "configuration_cloverlm.py": os.path.join(
            SCRIPT_DIR, "configuration_cloverlm.py"),
        "modeling_cloverlm.py": os.path.join(
            SCRIPT_DIR, "modeling_cloverlm.py"),
        "tokenization_cloverlm.py": os.path.join(
            SCRIPT_DIR, "tokenization_cloverlm.py"),
        "fake_quartet.py": os.path.join(SRC_DIR, "fake_quartet.py"),
    }

    # Files that need import patching
    files_to_patch = {
        "exp_transformer.py": os.path.join(
            SRC_DIR, "models", "transformer.py"),
        "exp_mlp.py": os.path.join(SRC_DIR, "models", "mlp.py"),
    }

    for dst_name, src_path in files_to_copy.items():
        dst_path = os.path.join(args.output_dir, dst_name)
        shutil.copy2(src_path, dst_path)
        print(f"  copied  {dst_name}")

    for dst_name, src_path in files_to_patch.items():
        with open(src_path) as f:
            src_text = f.read()
        patched = _patch_imports(src_text, dst_name)
        dst_path = os.path.join(args.output_dir, dst_name)
        with open(dst_path, "w") as f:
            f.write(patched)
        print(f"  patched {dst_name}")

    # ------------------------------------------------------------------
    print(f"\nDone.  Model saved to: {args.output_dir}")
    print(f"Load with:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained(")
    print(f'      "{args.output_dir}", trust_remote_code=True)')


if __name__ == "__main__":
    main()

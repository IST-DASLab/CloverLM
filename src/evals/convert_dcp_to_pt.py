
import argparse
import os
import sys
import time

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


MODEL_KEY_PREFIX = "checkpoint.model."


def discover_steps(checkpoint_dir):
    """Find all step subdirectories that contain a .metadata file."""
    steps = []
    for entry in sorted(os.listdir(checkpoint_dir)):
        full = os.path.join(checkpoint_dir, entry)
        if os.path.isdir(full) and entry.isdigit():
            if os.path.exists(os.path.join(full, ".metadata")):
                steps.append(entry)
    return steps


def convert_step(step_dir, output_path):
    """Load model weights from a DCP step directory and save as a flat .pt state_dict."""
    reader = FileSystemReader(step_dir)
    metadata = reader.read_metadata()

    # Pre-allocate tensors only for model parameters (skip optimizer state, scalars, etc.)
    state_dict = {}
    for fqn, meta in metadata.state_dict_metadata.items():
        if fqn.startswith(MODEL_KEY_PREFIX) and isinstance(meta, TensorStorageMetadata):
            state_dict[fqn] = torch.empty(meta.size, dtype=meta.properties.dtype)

    print(f"  Loading {len(state_dict)} model tensors from {len(os.listdir(step_dir)) - 1} shards ...")
    t0 = time.time()
    dcp.load(state_dict, checkpoint_id=step_dir, no_dist=True)
    load_time = time.time() - t0

    # Strip the DCP key prefix so the state_dict matches what eval.py expects
    model_sd = {}
    for fqn, tensor in state_dict.items():
        model_sd[fqn[len(MODEL_KEY_PREFIX):]] = tensor

    n_params = sum(v.numel() for v in model_sd.values())
    size_gb = sum(v.numel() * v.element_size() for v in model_sd.values()) / 1e9

    print(f"  Loaded in {load_time:.1f}s  —  {n_params:,} params ({n_params/1e9:.2f}B), {size_gb:.2f} GB")

    torch.save(model_sd, output_path)
    file_gb = os.path.getsize(output_path) / 1e9
    print(f"  Saved: {output_path}  ({file_gb:.2f} GB)")

    return model_sd


def main():
    p = argparse.ArgumentParser(
        description="Convert DCP distributed checkpoints to single .pt files for eval.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("checkpoint_dir", type=os.path.abspath,
                    help="Path to the *_checkpoint directory containing step subdirs")
    p.add_argument("output_dir", type=os.path.abspath,
                    help="Directory to write the converted .pt files")
    p.add_argument("--steps", nargs="*", default=None,
                    help="Specific steps to convert (default: all discovered steps)")
    p.add_argument("--name", default=None,
                    help="Base name for output files (default: derived from checkpoint_dir)")
    args = p.parse_args()

    base_name = args.name or os.path.basename(args.checkpoint_dir).replace("_checkpoint", "")

    if args.steps:
        steps = args.steps
    else:
        steps = discover_steps(args.checkpoint_dir)

    if not steps:
        print(f"ERROR: No valid step directories found in {args.checkpoint_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"  DCP → .pt  Converter")
    print(f"{'='*70}")
    print(f"  source : {args.checkpoint_dir}")
    print(f"  output : {args.output_dir}")
    print(f"  name   : {base_name}")
    print(f"  steps  : {steps}")
    print(f"{'='*70}\n")

    for step in steps:
        step_dir = os.path.join(args.checkpoint_dir, step)
        if not os.path.isdir(step_dir):
            print(f"SKIP: {step_dir} does not exist")
            continue

        output_path = os.path.join(args.output_dir, f"{base_name}-{step}.pt")
        print(f"[step {step}]")
        convert_step(step_dir, output_path)
        print()

    print(f"{'='*70}")
    print(f"  All conversions complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

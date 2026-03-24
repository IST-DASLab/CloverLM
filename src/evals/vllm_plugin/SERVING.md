# Serving CloverLM with vLLM (Quartet II NVFP4)

## Prerequisites

- NVIDIA Blackwell GPU (B300 / B200 / RTX 5090) for real Quartet II NVFP4 kernels
- CUDA 13.0+
- Python 3.11+
- The Quartet II kernels (`quartet2` package) installed

## 1. Environment Setup

```bash
# Activate the existing environment
source .venv/bin/activate

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda-13.0/
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
```

## 2. Install vLLM

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest \
    | jq -r .tag_name | sed 's/^v//')
export CUDA_VERSION=130
export CPU_ARCH=$(uname -m)

uv pip install \
    "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_35_${CPU_ARCH}.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
```

## 3. Serve the Model

### Offline inference (quick test)

```bash
cd /home/matin/convert_dir/CloverLM/vllm_plugin
python serve.py
```

### OpenAI-compatible API server

```bash
cd /home/matin/convert_dir/CloverLM/vllm_plugin
python serve.py --api --port 8000
```

Then query:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/matin/convert_dir/CloverLM",
        "prompt": "The capital of France is",
        "max_tokens": 64,
        "temperature": 0.8
    }'
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `../` (CloverLM dir) | Path to CloverLM model directory |
| `--api` | off | Start OpenAI-compatible API server |
| `--port` | 8000 | API server port |
| `--host` | 0.0.0.0 | API server host |
| `--tp` | 1 | Tensor parallel size |
| `--max-model-len` | 1024 | Maximum context length |
| `--gpu-memory-utilization` | 0.9 | GPU memory fraction to use |

## Architecture

The vLLM integration consists of three components:

1. **`quartet2_quant.py`** -- Quartet II quantization plugin registered as `"quartet2"`.
   Wraps the Quartet II on-the-fly FP4 quantization (`quant_fp4` + `flashinfer.mm_fp4`)
   into vLLM's `LinearMethodBase` interface. Weights stay in bf16; quantization happens
   at each forward pass.

2. **`cloverlm_vllm.py`** -- Full vLLM model implementation with paged KV cache.
   Reimplements CloverLM's architecture using vLLM primitives:
   - `ColumnParallelLinear` / `RowParallelLinear` for Q/K/V/O and MLP projections
   - vLLM `Attention` for paged KV caching and efficient attention
   - Custom RoPE (base 1024, repeat_interleave pattern)
   - Sphere normalization on Q/K before attention
   - Per-head learnable scale parameter
   - Squared ReLU activation in MLP
   - Post-sublayer RMSNorm (not pre-norm)

3. **`serve.py`** -- Entry point that registers both the quantization plugin and model,
   then launches vLLM in offline or API mode.

## Known Limitations

- **CUDA graphs**: Currently `enforce_eager=True` is required because the Quartet II
  on-the-fly quantization kernels (`quant_fp4` + `mm_fp4`) are not compatible with
  CUDA graph capture. This means slightly higher per-token latency compared to
  CUDA-graph-enabled models. A future update to the Quartet II kernels could remove
  this limitation.

## Troubleshooting

**"No module named 'quartet2'"**: Ensure the Quartet II kernels are installed:
```bash
uv pip install "quartet2 @ git+https://github.com/IST-DASLab/Quartet-II.git#subdirectory=kernels"
```

**CUDA errors**: Make sure `CUDA_HOME` points to CUDA 13.0+ and `TRITON_PTXAS_PATH` is set.

**Out of memory**: Reduce `--gpu-memory-utilization` or use `--tp 2` for tensor parallelism.

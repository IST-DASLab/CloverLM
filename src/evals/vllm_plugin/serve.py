

import argparse
import os
import sys

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(PLUGIN_DIR)
sys.path.insert(0, PLUGIN_DIR)

# Register the Quartet II quantization plugin before any vLLM imports
import quartet2_quant  # noqa: F401  — triggers @register_quantization_config

from vllm import ModelRegistry
from cloverlm_vllm import CloverLMForCausalLM_vLLM

ModelRegistry.register_model(
    "CloverLMForCausalLM", CloverLMForCausalLM_vLLM,
)


def main():
    parser = argparse.ArgumentParser(description="Serve CloverLM with vLLM")
    parser.add_argument(
        "--model", default=MODEL_DIR,
        help="Path to CloverLM model directory",
    )
    parser.add_argument("--api", action="store_true", help="Start OpenAI API server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--max-model-len", type=int, default=1024,
        help="Maximum context length",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
    )
    args = parser.parse_args()

    if args.api:
        _serve_api(args)
    else:
        _offline_inference(args)


def _offline_inference(args):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        quantization="quartet2",
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=128,
    )

    prompts = [
        "The capital of France is",
        "Large language models are",
        "In the year 2030,",
    ]

    print("=" * 60)
    print("  CloverLM — vLLM Offline Inference (Quartet II NVFP4)")
    print("=" * 60)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"\nPrompt:    {prompt}")
        print(f"Generated: {generated}")


def _serve_api(args):
    sys.argv = [
        "vllm",
        "--model", args.model,
        "--quantization", "quartet2",
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", str(args.tp),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enforce-eager",
        "--host", args.host,
        "--port", str(args.port),
    ]
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.entrypoints.openai.api_server import run_server
    import asyncio

    vllm_parser = make_arg_parser(FlexibleArgumentParser())
    vllm_args = vllm_parser.parse_args()
    asyncio.run(run_server(vllm_args))


if __name__ == "__main__":
    main()

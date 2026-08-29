"""Minimal canonical CompilerKV invocation on already collected prefill tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kv_compression import CompilerKVCompressor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tables", type=Path, help="directory produced by compilerkv-compile")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--tensor-file", type=Path, required=True)
    args = parser.parse_args()

    tensors = torch.load(args.tensor_file, map_location="cpu", weights_only=True)
    required = {"attention", "key_cache", "value_cache", "token_logprobs"}
    missing = required - tensors.keys()
    if missing:
        raise ValueError(f"tensor file is missing: {sorted(missing)}")

    compressor = CompilerKVCompressor(args.tables, observation_window=64)
    output = compressor.compress(
        attention=tensors["attention"],
        key_cache=tensors["key_cache"],
        value_cache=tensors["value_cache"],
        token_logprobs=tensors["token_logprobs"],
        budgets=args.budget,
        token_mask=tensors.get("token_mask"),
        query_mask=tensors.get("query_mask"),
    )
    retained = [[len(sample) for sample in layer] for layer in output.indices]
    print(f"retained tokens per layer/batch: {retained}")
    print(f"attention entropy: {output.attention_entropy.tolist()}")
    print(f"local PPL: {output.local_perplexity.tolist()}")


if __name__ == "__main__":
    main()

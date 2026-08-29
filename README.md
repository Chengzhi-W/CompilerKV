# CompilerKV

[![Paper](https://img.shields.io/badge/arXiv-2602.08686-b31b1b.svg)](https://arxiv.org/abs/2602.08686)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](CompilerKV_v1/pyproject.toml)

Paper-aligned implementation of
**[CompilerKV: Risk-Adaptive KV Compression via Offline Experience
Compilation](https://arxiv.org/abs/2602.08686)** (arXiv:2602.08686).

CompilerKV makes one irreversible retention decision at the end of prefill and
keeps the resulting cache fixed during decoding. The online policy contains no
learning: it scans the final observation-window attention rows, performs two
compiled table lookups per layer, applies an elastic threshold, and clamps only
candidate sets that exceed the layer budget.

## What is implemented

The canonical implementation is in `CompilerKV_v1/kv_compression/core.py` and
follows the paper's equations and Algorithm 1:

1. **Stabilized utility (E1)**

   `alpha_t = (T / W) * sum_{j in Omega} mean_{l,h}(A[l,h,j,t])`

   `rho[l,h,t] = ||V[l,h,t]||_2 / mean_k ||V[l,h,k]||_2`

   `u[l,h,t] = alpha_t * rho[l,h,t]`

2. **Head heterogeneity (E2)**

   `score[l,t] = max_h u[l,h,t] * W_head[l,h,B_l]`

   The max is tokenwise. A retrieval-critical head can therefore veto noisy
   heads; the implementation never collapses `W_head` to a single layer scalar
   and never averages the weighted heads.

3. **Risk-adaptive gate (E3)**

   Attention entropy and local observation-window perplexity are discretized
   with calibration-derived edges. `T_gate[l,b_ent,b_ppl,B_l]` returns an
   absolute threshold in `[0.8, 1.0]`.

4. **Elastic selection**

   All tokens whose absolute score reaches the threshold are retained when the
   candidate set is under budget. Only overflow is reduced with Top-`B_l`.
   Candidate sets are not padded to the budget, scores are not max-normalized,
   and the observation window is not automatically added to the retained set.

The offline compiler in `CompilerKV_v1/kv_compression/compiler/` implements the
horizon-1 CQL objective shared by both tables. It uses the paper defaults:
AdamW, learning rate `3e-4`, batch size `4096`, 10 epochs, conservative weight
`0.75`, 29 head actions in `[0.8,1.5]`, 21 threshold actions in `[0.8,1.0]`, and
a `20 x 4` entropy-perplexity grid. Rewards are
`-(L_comp-L_full)` plus the gate's asymmetric over-retention penalty.

## Repository layout

```text
CompilerKV_v1/
  kv_compression/
    core.py                 # canonical online operator
    api.py                  # public prefill-only API
    compiler/               # record schema + horizon-1 CQL compiler
    token_drop/             # legacy Transformers monkeypatch integration
  tables/                   # schema/docs; real compiled tables go here
  tests/                    # CPU unit and regression tests
  run/longbench/            # LongBench harness
```

## Install and test

```bash
cd CompilerKV_v1
python -m pip install -e '.[test]'
pytest
```

The optional model integration is pinned to the Transformers API used by this
codebase:

```bash
python -m pip install -e '.[evaluation]'
```

## Compile the offline tables

The repository intentionally does not ship the old simulated tables as learned
artifacts. Collect state-action records on a held-out, evaluation-disjoint
calibration corpus, with each table's action evaluated while the other current
table is active. See
`CompilerKV_v1/tables/calibration_record.example.jsonl` for the JSONL schema and
`CompilerKV_v1/docs/calibration.md` for the rollout/alternation protocol.

```bash
cd CompilerKV_v1
compilerkv-compile /path/to/calibration.jsonl tables/compiled/llama3-8b \
  --num-layers 32 \
  --num-heads 32
```

The output is `W_head.npy`, `T_gate.npy`, and a provenance-rich
`manifest.json`. Budget-conditioned tables use shapes `[N_B,L,H]` and
`[N_B,L,20,4]`. Single-budget tables are also accepted. Runtime depth is mapped
by relative-depth interpolation, as described in the paper. Budget-conditioned
artifacts require an exact compiled budget by default, and head counts must
match; opt-in interpolation flags can be recorded in the manifest for explicit
cross-architecture experiments.

## Run the canonical prefill operator

```python
from kv_compression import CompilerKVCompressor

compressor = CompilerKVCompressor(
    tables="tables/compiled/llama3-8b",
    observation_window=64,
)
output = compressor.compress(
    attention=attention,          # [L,B,H,W,T] or [L,B,H,T,T]
    key_cache=key_cache,          # [L,B,H_kv,T,D]
    value_cache=value_cache,      # [L,B,H_kv,T,D]
    token_logprobs=token_logp,    # [B,T], for local PPL
    budgets=512,
)

# Exact dense cache for the standard batch-1 decoding path.
compressed_keys, compressed_values = output.dense_cache_for_batch_one()
```

For padded batches, pass `token_mask` and `query_mask`. The result remains
ragged by design because elastic under-retention may retain a different number
of tokens for each prompt.

## Transformers monkeypatch

The historical LLaMA/Mistral/Qwen2/InternLM monkeypatches remain available for
compatibility. They now use the correct per-layer score and elastic selector,
but a layer-local attention forward cannot observe the paper's cross-layer
attention mean or final LM-head token probabilities. For paper-faithful results,
use the canonical end-of-prefill API and provide `token_logprobs`. If the legacy
adapter is used, set `attention.config.prompt_ppl` explicitly; otherwise it emits
a warning and uses the compiled median PPL bin.

## Paper settings and reported results

The submitted paper compiles on about 50K long-context prompts disjoint from
LongBench, with a 64-query observation window and a 128-token continuation loss.
At a 512-token per-layer budget, the reported LongBench averages are 42.61
(InternLM2.5-7B), 42.55 (LLaMA-3-8B), 41.13 (Qwen2-7B), and 42.94
(Mistral-7B), for a four-backbone mean of 42.31. These numbers require the real
calibration corpus, compiled tables, model checkpoints, and evaluation assets;
the repository does not fabricate those artifacts.

## Citation

```bibtex
@misc{compilerkv2026,
  title         = {CompilerKV: Risk-Adaptive KV Compression via Offline Experience Compilation},
  author        = {Yang, Ning and Wang, Chengzhi and Liu, Yibo and Tian, Baoliang and Zhang, Haijun},
  year          = {2026},
  eprint        = {2602.08686},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2602.08686}
}
```

# Offline calibration protocol

This document describes the inputs expected by `compilerkv-compile`. It mirrors
Section C.7 of the submitted paper and separates expensive model rollouts from
the lightweight CQL table fit.

## Corpus split

Use a held-out corpus that is strictly disjoint from every evaluation prompt.
The paper uses about 50K unlabeled long-context prompts spanning long-form
narratives, scientific and technical documents, reports and transcripts,
dialogue/instruction data, code, and QA text. Public-source examples named in
the paper include PG19, arXiv, PubMed, GovReport, QMSum, BookSum,
ShareGPT/UltraChat, and GitHub/StackExchange subsets of The Pile.

The compiler needs no task labels. Each prompt contributes prefill attention
statistics plus a 128-token continuation loss. Use a 64-query observation
window unless running an explicit ablation.

## Risk coverage

For every prompt, compute:

- structural risk: entropy of the normalized, layer/head-averaged
  observation-window attention mass;
- semantic risk: perplexity over the same observation window.

Build the experience corpus with stratified coverage over the joint risk
coordinates. The compiler derives quantile edges and balances observed states,
but that cannot recover high-risk regions that were never collected.

## Coupled alternating rollouts

The runtime representation is factorized, while rewards are coupled:

1. Hold the current gate table fixed. For each `(layer, head, budget)` state,
   perturb one head weight over the 29-action grid `[0.8,1.5]`, run the complete
   retention pipeline (including the gate and budget clamp), and record the
   continuation loss. Set `paired_policy` to the gate-table round or checksum.
2. Hold the updated head table fixed. For each
   `(layer, entropy_bin, ppl_bin, budget)` state, perturb the threshold over the
   21-action grid `[0.8,1.0]`, run the complete pipeline, and record loss and
   candidate count. Set `paired_policy` to the head-table round or checksum.
3. Alternate the two rollout/compile phases until table actions stop changing
   or the chosen validation criterion converges.

The compiler rejects missing `paired_policy` values by default. Use
`--allow-independent-records` only to reproduce the independent-compilation
ablation.

## JSONL fields

Every line contains one observed state-action tuple:

| Field | Head | Gate | Meaning |
|---|---:|---:|---|
| `kind` | yes | yes | `head` or `gate` |
| `prompt_id` | yes | yes | stable calibration example id |
| `layer` | yes | yes | zero-based layer |
| `head` | yes | no | zero-based attention head |
| `budget` | yes | yes | total retained tokens for the layer |
| `action` | yes | yes | action-grid value |
| `full_loss` | yes | yes | continuation NLL with FullKV |
| `compressed_loss` | yes | yes | continuation NLL under this action |
| `sequence_length` | yes | yes | prefill token count `T` |
| `entropy` | no | yes | structural risk before discretization |
| `ppl` | no | yes | local observation-window perplexity |
| `candidate_count` | no | yes | candidates before Top-`B` clamp |
| `paired_policy` | yes | yes | opposite-table round/checksum |

The compiler constructs the reward as
`-(compressed_loss-full_loss)`. Gate records additionally subtract
`max(0,candidate_count-budget)/sequence_length`; head records receive no budget
penalty. An explicit `reward` may be supplied for diagnostics, but normal
paper-aligned runs should retain the underlying losses and counts for audit.

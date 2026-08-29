# Local table output (not distributed)

This is a code-only release: this directory contains no learned, simulated, or
hand-shaped policy tables. The empty `compiled/` directory is only a convenient
local output location. Generate model-specific artifacts from your own
calibration experience with the offline compiler; the runtime accepts only
artifacts accompanied by the compiler-generated `manifest.json`.

Compile coupled calibration experience with the paper defaults:

```bash
compilerkv-compile calibration.jsonl tables/compiled/model-name \
  --num-layers 32 --num-heads 32
```

Your local output will contain:

- `W_head.npy`: `[budget, layer, head]`, with actions in `[0.8, 1.5]`;
- `T_gate.npy`: `[budget, layer, entropy_bin, ppl_bin]`, with 20 x 4 risk bins
  and actions in `[0.8, 1.0]`;
- `manifest.json`: risk-bin edges, budgets, CQL hyperparameters, support counts,
  reward definition, and input checksum.

The JSONL schema is illustrated by `calibration_record.example.jsonl`. A head
record must evaluate the perturbed head action while the current risk gate is
active. A gate record must evaluate its threshold while the current head table
is active. The `paired_policy` field records this coupled reward provenance.

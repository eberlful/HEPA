# HEPA

Minimal PyTorch implementation of the HEPA training recipe from
`paper/HEPA.pdf`.

The script runs the two phases described in the paper:

1. Self-supervised JEPA pretraining with a causal patch Transformer encoder,
   a horizon-conditioned predictor, L1 latent prediction loss, and SIGReg.
2. Supervised predictor finetuning with the encoder frozen and a discrete
   hazard head composed into a monotone survival CDF over horizons.

Run a quick CPU smoke test with synthetic event data:

```bash
.venv/bin/python main.py \
  --pretrain-epochs 1 \
  --finetune-epochs 1 \
  --model-dim 32 \
  --heads 4 \
  --train-samples 32 \
  --val-samples 16 \
  --batch-size 8 \
  --context-length 64 \
  --horizons 16 \
  --series-length 160
```

For paper-like model size, keep the defaults for `--model-dim`, `--layers`,
`--heads`, and `--patch-size`, then increase epochs and samples for a real
dataset.

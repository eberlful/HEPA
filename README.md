# HEPA

Paper-oriented PyTorch training harness for the HEPA recipe in
`paper/hepa.md` / `paper/HEPA.pdf`.

The canonical entrypoint is `main.py`, which delegates to
`hepa.train_hepa`. It implements:

1. Self-supervised JEPA pretraining with a causal patch Transformer encoder,
   weight-shared bidirectional target path, horizon-conditioned predictor,
   L1 latent prediction loss, and SIGReg.
2. Predictor finetuning with the encoder frozen, positive-weighted BCE, and
   a monotone discrete survival CDF over dense horizons.

## Run

Quick CPU smoke test:

```bash
.venv/bin/python main.py --data mock --preset smoke --device cpu
```

Paper-like mock run:

```bash
.venv/bin/python main.py --data mock --preset paper --device auto
```

Useful overrides:

```bash
.venv/bin/python main.py \
  --data mock \
  --preset smoke \
  --seed 1 \
  --channels 8 \
  --context-len 96 \
  --max-horizon 24 \
  --pretrain-epochs 2 \
  --finetune-epochs 2 \
  --batch-size 8 \
  --device cpu \
  --no-checkpoint
```

## Dataset Contract

Real datasets should produce `EventBatch` objects with:

- `context`: `(B, T, S)` past observations.
- `future`: `(B, K, S)` future observations for target representations.
- `horizons`: `(K,)` dense integer horizons `1..K`.
- `labels`: `(B, K)` cumulative labels where `labels[:, k]` is
  `1[event occurs within horizon k + 1]`.

The first implemented data source is `MockEventDataset`, which exercises the
full pretraining, finetuning, h-AUROC, and monotonicity path without requiring
benchmark downloads.

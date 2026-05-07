# apps/baselines/clip/

Contrastive trainer + feature extractor for CLIP-style image-text alignment
models. The trainer learns a shared embedding space for a vision tower
(DINOv2 or ViT) and a text tower (BERT) using InfoNCE, optionally
co-trained with masked language modeling on a separate text corpus and/or
DINOv2 self-supervised learning on the same images.

## Trainer modes

| Mode | Losses | Notes |
|---|---|---|
| `contrastive`      | InfoNCE                       | Vanilla CLIP-style; one optimizer. |
| `interleaved_lm`   | InfoNCE + BERT MLM            | Alternates contrastive batches with MLM batches drawn from a separate text-only corpus. |
| `interleaved_dino` | InfoNCE + DINOv2 SSL          | Alternates contrastive batches with DINOv2 SSL on the same images; teacher backbone is copied into the CLIP vision encoder after each SSL block. |
| `triple`           | InfoNCE + BERT MLM + DINOv2   | Round-robin all three. |

Schedule per mode is set under `mode.interleave`; e.g.
`mode.interleave={contrastive: 4, mlm: 1}` runs four contrastive steps then
one MLM step, and repeats.

## Layout

```
apps/baselines/clip/
  configs/
    config.yaml          # default composition (mode=contrastive)
    mode/                # per-mode interleave schedules
    model/                 # default model composition (embedding_dim, temperature, etc.)
    text_encoder/          # bert_base.yaml: BERT-base TextEncoder
    vision_encoder/        # hub_dinov2_*.yaml, custom_dinov2.yaml, random_vit_*.yaml
    data/                # coco / ego4d / howto manifest schemas
    optim/               # AdamW + cosine schedule
    text_only_data/      # text-only corpus for the MLM head
    dinov2/              # bundled DINOv2 SSL config
    checkpoint/, wandb/  # output + tracking
  modeling/              # text encoder, vision encoder, multimodal model, MLM head, DINOv2 SSL wrapper
  data/                  # caption datasets, text-only dataset, transforms, collate
  training/              # trainer loop, optimizer factories, interleave scheduler, checkpoint, W&B
  scripts/               # checkpoint conversion utilities
  extractor.py           # downstream feature extractor (already shipped)
```

## Quickstart

### Single-GPU contrastive on COCO

```bash
egobabyvlm-train-contrastive \
    name=coco_baseline \
    data.train_dataset.manifest_path=/data/coco/preprocessed_captions_train.json \
    data.train_dataset.image_root=/data/coco/all_images \
    data.val_dataset.manifest_path=/data/coco/preprocessed_captions_val.json \
    checkpoint.save_dir=$HOME/runs/coco_baseline
```

### Multi-GPU triple mode

```bash
torchrun --standalone --nproc-per-node=4 \
    -m apps.baselines.clip.training.train \
    name=coco_triple mode=triple \
    data.train_dataset.manifest_path=/data/coco/preprocessed_captions_train.json \
    data.train_dataset.image_root=/data/coco/all_images \
    data.val_dataset.manifest_path=/data/coco/preprocessed_captions_val.json \
    +text_only_data=default text_only_data.train_file=/data/text/captions.txt \
    +dinov2=vitb14 \
    checkpoint.save_dir=$HOME/runs/coco_triple
```

### Resume from a checkpoint

```bash
egobabyvlm-train-contrastive \
    name=coco_baseline \
    ... \
    checkpoint.resume_from=$HOME/runs/coco_baseline/checkpoints/latest.pt
```

## Checkpoint format

Each checkpoint is a single `.pt` file containing the multimodal model
state_dict, optionally an MLM head + DINOv2 SSL state, all optimizer and
scheduler state, the resolved Hydra config, and metadata (epoch, step, best
validation loss). Loading does **not** require any sidecar config files —
the `config` is embedded verbatim and the feature extractor instantiates
encoders from it.

## Migrating older Lightning checkpoints

Older `MultiModalLitModel` `.ckpt` files (with a vocab → string → BERT
roundtrip in the text encoder) can be converted to the self-contained
format used here:

```bash
egobabyvlm-convert-legacy-ckpt \
    --legacy-ckpt /old/run/checkpoints/best.ckpt \
    --output /old/run/checkpoints/best.pt \
    --hf-model-name bert-base-uncased \
    --vision-model-name dinov2_vitb14 \
    --embedding-dim 512 \
    --normalize-features
```

The converter strips Lightning prefixes and the vocab fallback layer,
then writes the self-contained format with the original HF model name
pinned into the embedded config. Once converted, the resulting `.pt`
loads through the same feature extractor as a freshly trained model.

## Datasets supported

COCO (Karpathy split), Ego4D, and HowTo100M ship as concrete instantiations.
Adding a new caption dataset is one Python class + one YAML — see
`apps/baselines/clip/data/captions.py`.

## Documentation

See `docs/contrastive_training.md` for an end-to-end walkthrough
including data preparation and common config overrides.

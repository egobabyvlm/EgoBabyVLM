# Contrastive training

`apps/baselines/clip/training/` provides a single Hydra-driven trainer
that covers four CLIP-style contrastive recipes used in the EgoBabyVLM
paper: vanilla contrastive, contrastive + BERT MLM, contrastive +
DINOv2 SSL, and the triple combination of all three.

## Modes

| Mode | When to use |
|---|---|
| `contrastive`      | You only need image-text alignment. Cheapest, fastest. |
| `interleaved_lm`   | The text tower is BERT and you have an external text corpus you want to keep training on (prevents catastrophic forgetting of the LM). |
| `interleaved_dino` | The vision tower is DINOv2 and you want to keep training the SSL objective on the same images, then sync the teacher backbone into the contrastive vision encoder. |
| `triple`           | Combine MLM + SSL on top of contrastive. |

The mode selects which loss heads are active and how often. The actual
schedule is `mode.interleave`:

```yaml
# mode/interleaved_lm.yaml
interleave:
  contrastive: 10  # 10 contrastive steps...
  mlm: 1           # ...then 1 MLM step.
```

## Data

Two manifest formats are supported:

- **COCO Karpathy** (`*.json`): top-level `images` array, each item has
  `filename` and `sentences: [{raw, tokens}]`.
- **HowTo100M / Ego4D / BabyView** (`*.json`): list of dicts with
  `utterance` and `frame_filenames`.

The text-only MLM corpus is plain text, one example per line (override
`text_only_data.train_file`).

## Quickstart

### Single-GPU contrastive on COCO

```bash
egobabyvlm-train-contrastive \
    name=coco_baseline \
    data.train_dataset.manifest_path=/data/coco/preprocessed_captions_train.json \
    data.train_dataset.image_root=/data/coco/all_images \
    data.val_dataset.manifest_path=/data/coco/preprocessed_captions_val.json \
    epochs=10 \
    optim.lr=3e-4 \
    checkpoint.save_dir=$HOME/runs/coco_baseline
```

### Multi-GPU triple mode on Ego4D

```bash
torchrun --standalone --nproc-per-node=4 \
    -m apps.baselines.clip.training.train \
    name=ego4d_triple mode=triple \
    data=ego4d \
    data.train_dataset.manifest_path=/data/ego4d/train.json \
    data.train_dataset.image_root=/data/ego4d/frames_1fps \
    data.val_dataset.manifest_path=/data/ego4d/val.json \
    +text_only_data=default \
    text_only_data.train_file=/data/ego4d/narrations.txt \
    +dinov2=vitb14 \
    epochs=20 \
    checkpoint.save_dir=$HOME/runs/ego4d_triple
```

## Output layout

The default checkpoint policy writes:

```
$checkpoint.save_dir/
  latest.pt       # last completed step
  epoch_0000.pt   # one per epoch
  epoch_0001.pt
  ...
  best.pt         # best validation loss
```

Set `checkpoint.keep_last=N` to retain only the most recent N
`epoch_*` files.

## Loading a trained checkpoint

```python
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from apps.baselines.clip.modeling import MultiModalModel

payload = torch.load("/path/to/checkpoint.pt", weights_only=False)
cfg = OmegaConf.create(payload["config"])

# The embedded config carries the full _target_ for both encoders, so
# hydra.utils.instantiate dispatches to whichever vision encoder class
# (Hub / custom DINOv2 / random ViT) was used at train time.
text_encoder = instantiate(cfg.model.text_encoder)
vision_encoder = instantiate(cfg.model.vision_encoder)
model = MultiModalModel(
    vision_encoder, text_encoder,
    normalize_features=cfg.model.normalize_features,
    temperature=cfg.model.temperature,
    fix_temperature=cfg.model.fix_temperature,
)
model.load_state_dict(payload["model_state_dict"])
model.eval()
```

## Bundled DINOv2

The DINOv2 SSL implementation is under
`apps/baselines/dinov2/third_party/dinov2/`. See that directory's
`README.md` for the upstream provenance and refresh procedure.

# apps/alignment_scoring/

Pipelines for measuring image/video–text alignment with off-the-shelf and
fine-tuned models. Used to filter and re-caption the contrastive training
data described in the EgoBabyVLM paper.

## Pipelines

| Pipeline | Model family | What it does |
|---|---|---|
| `alignment-clip-scoring` | CLIP / [Perception Encoder](https://arxiv.org/abs/2504.13181) | Score matched-vs-shuffled (image, caption) pairs with cosine similarity; bootstrap JS divergence between the two distributions |
| `alignment-sts-scoring` | [SONAR](https://arxiv.org/abs/2308.11466) | Same matched-vs-shuffled aggregation, but on text-only pairs encoded by SONAR (e.g. originals vs PLM re-captions) |
| `alignment-captioning` | [Perception-LM](https://arxiv.org/abs/2504.13180) | Re-caption a manifest with PLM and write the result back into a manifest copy |
| `alignment-vqa-scoring` | Perception-LM | Score matched-vs-shuffled pairs by P("Yes" \| "Does this figure show '<caption>'?"); bootstrap JS like clip / sts scoring |

## Layout

```
apps/alignment_scoring/
  configs/
    pipeline/    # one YAML per pipeline (clip_scoring, sts_scoring, ...)
    dataset/     # CocoCaptionsDataset, KarpathyCocoCaptionsDataset, VideoCaptionsDataset
    dataset_path/# path-only variants (for STS, captioning, VQA)
    model/       # ViT-B/16, PE-Core-bigG-14-448, PLM 1B/8B
  data/          # caption datasets (COCO, video CSV, windowed video, text-pair)
  modeling/      # PLMGenerationModule + PackedCausalTransformerGenerator
  pipelines/     # one Hydra entrypoint per pipeline
  scripts/       # manifest tooling (shuffle, etc.)
  third_party/   # perception_models subset under FAIR Noncommercial Research License
  configs.py     # all Hydra dataclass schemas
  utils.py       # open_clip helpers, AdamW + grad scaler, JS / KL stats
```

## Adding a new dataset

1. Write a `CaptionsMediaDataset` subclass in `apps/alignment_scoring/data/`
   that returns `(media, text, media_id)` triples.
2. Add a YAML to `configs/dataset/<name>.yaml` pointing `_target_` at the
   new class.
3. (For STS / captioning / VQA) add a path-only variant returning
   `(media_path, text, media_id)` and a `configs/dataset_path/<name>.yaml`.

No pipeline changes needed.

## Quickstart

```bash
# Install (provides the alignment-* CLIs)
pixi install -e dev

# Build a shuffled manifest for matched-vs-shuffled scoring
alignment-create-shuffled-manifest \
  --manifest-path /data/coco/captions_train2017.json \
  --output-path /data/coco/captions_train2017_shuffled.json \
  --type json

# Run CLIP scoring locally on small data to sanity check
alignment-clip-scoring \
  --config-path apps/alignment_scoring/configs \
  --config-name pipeline/clip_scoring \
  name=coco_smoke \
  matched_processor.data.dataset.manifest_path=/data/coco/captions_train2017.json \
  matched_processor.data.dataset.dataset_dir=/data/coco/train2017 \
  shuffled_processor.data.dataset.manifest_path=/data/coco/captions_train2017_shuffled.json \
  shuffled_processor.data.dataset.dataset_dir=/data/coco/train2017 \
  model@matched_processor.model=vit_b16_openai \
  model@shuffled_processor.model=vit_b16_openai
```

To run on SLURM, override `launcher.cluster=slurm` and add `launcher.update_parameters.slurm_qos=...`.

## Tests

- Unit tests: `pixi run -e dev pytest -q tests/apps/alignment_scoring/`
- GPU smoke tests: `pixi run -e dev pytest -m gpu tests/apps/alignment_scoring/`
  (Requires CUDA + downloads ViT-B/16 + SONAR-text on first run.)

## perception_models

`third_party/perception_models/` is a namespaced subset of [FAIR's
perception_models](https://github.com/facebookresearch/perception_models)
needed for PLM captioning + VQA scoring. See
`third_party/perception_models/README.md` for what's there, why, and
the refresh script. Released under FAIR Noncommercial Research License.

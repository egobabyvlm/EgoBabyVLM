# EgoBabyVLM

Reference implementation for the **EgoBabyVLM Challenge**: training
vision-language models (VLMs) using only naturalistic infant egocentric
video, and evaluating them under a fixed multimodal + unimodal probe
suite.

> *Anonymous code drop accompanying a paper currently under review.*

## The Challenge

Train a VLM on the [BabyView 2025.1](https://databrary.org/volumes/1882)
corpus (≈863h of head-mounted-camera video from children) **and nothing
else** — no extra image, video, text, or audio data may be used for any
encoder pretraining, fine-tuning, or evaluation. The challenge is to
beat the chance-level baselines on a fixed evaluation suite without
relying on web-scale priors.

Submissions are scored on three families of tasks, each with subgroup
aggregates and an overall:

| Family | Subgroups | Tasks |
|---|---|---|
| **Cross-modal grounding** ([Machine-DevBench](apps/benchmark_creation/README.md)) | Lexical (2), Grammatical (8) | Noun + adjective recognition; subject-verb / subject-adjective binding; negation; word order; prepositions; comparatives; counting; embedded relatives. ~3,700 contrastive (image, caption) trials sampled from the model's own training vocabulary across log-frequency bins. |
| **Vision** | Object recognition (6), Visual properties (3) | ImageNet-1k (k-NN, linear, ABX); MNIST (linear, ABX); COCO-Stuff segmentation; NYUv2 depth; CountBench (linear, ABX). Reported as *delta* vs the same-data unimodal DINOv2 baseline. |
| **Language** | Syntax (3), Semantics (2) | Zorro; LongTail-Swap (Inflection, Agreement, Word); Visual-Property Swap (color, material, size, shape). |

The paper shows weakly-aligned naturalistic egocentric input drives
contrastive and generative reference baselines to near-chance on the
cross-modal probes, while curated captions (COCO) approach off-the-shelf
CLIP. The challenge is to close that gap algorithmically, without
changing the data.

## Repository layout

```
apps/
├── data_preprocessing/   # video → frames + WhisperX transcripts + train/val/test manifests
├── baselines/
│   ├── dinov2/           # DINOv2 SSL + ViT-B/14 feature extractor
│   ├── lm_training/      # BERT MLM and GPT-2 from-scratch trainers
│   ├── clip/             # CLIP+ contrastive trainer (4 modes: contrastive, +MLM, +DINOv2, triple)
│   └── llava/            # EgoBabyLLaVA generative VLM trainer
├── alignment_scoring/    # CLIP / VQA / STS / captioning pipelines for cross-modal alignment scoring
├── benchmark_creation/   # Machine-DevBench corpus-grounded benchmark generator
└── swapbench/            # LongTail-Swap + VP-Swap generators
core/                     # Protocols (FeatureExtractor, …), DDP / config / seed utils
evaluation/               # Hydra+Stopes eval pipeline (vision, text, multimodal task launchers)
docs/                     # Per-component design notes
scripts/eval_data/        # Eval-dataset download helpers
tests/                    # Unit + opt-in integration tests
```

Each directory under `apps/` has a README with component-specific usage.

## A submission, end to end

1. **Preprocess raw video** — `apps/data_preprocessing/` extracts frames
   at 1 FPS, transcribes audio with WhisperX, optionally drops key-child
   speech with a VTC filter (BabyView), and builds a `(frame, utterance)`
   manifest.
2. **Pretrain unimodal encoders** — `apps/baselines/dinov2/` (vision
   SSL) and `apps/baselines/lm_training/` (BERT MLM and/or GPT-2 from
   scratch).
3. **Train the multimodal model** — `apps/baselines/clip/` (contrastive)
   or `apps/baselines/llava/` (generative).
4. **Evaluate** — `evaluation/` runs the three eval families and writes
   subgroup + overall aggregates.

`apps/alignment_scoring/`, `apps/benchmark_creation/`, and
`apps/swapbench/` are toolkits used to produce the paper's analyses and
benchmarks; they are not on the critical path for a challenge submission
but are useful for re-running those experiments or generating benchmarks
for a new training corpus.

## Install

The full environment is pinned in [`pixi.toml`](pixi.toml) (Python 3.12,
PyTorch 2.8 + CUDA 12.6, the heavy ML stack, dev tooling).

```bash
# install pixi: https://pixi.sh/latest/installation/
pixi install -e dev
```

This produces installable CLI entry points; the most commonly used:

```text
egobabyvlm-extract-frames                 # apps/data_preprocessing/frames
egobabyvlm-transcribe-whisperx            # apps/data_preprocessing/transcription
egobabyvlm-filter-vtc                     # apps/data_preprocessing/transcription (BabyView KCHI)
egobabyvlm-build-clip-manifest            # apps/data_preprocessing/manifests
egobabyvlm-train-contrastive              # apps/baselines/clip/training
egobabyvlm-swapbench-build-word-lists     # apps/swapbench (LongTail-Swap + VP-Swap)
egobabyvlm-swapbench-lt-swap              # apps/swapbench
egobabyvlm-swapbench-vp-swap              # apps/swapbench
alignment-{clip,sts,vqa}-scoring          # apps/alignment_scoring
alignment-captioning                      # apps/alignment_scoring
```

DINOv2, LLaVA, BERT, GPT-2, Machine-DevBench generation, and the
evaluation launcher are invoked via `python -m apps.<…>` /
`python -m evaluation.…` (see the corresponding component READMEs).

## Quickstart

### Run an evaluation

Download the eval datasets once (see
[`docs/eval_data.md`](docs/eval_data.md)), then point the launcher at
your model:

```bash
# All vision tasks
python -m evaluation.eval_launcher \
    eval=vision/vision_pipeline \
    model=dino \
    name=my_run

# All Machine-DevBench tasks (realistic + cartoon styles)
python -m evaluation.eval_launcher \
    eval=multimodal/machine_devbench_pipeline \
    model=clip_image \
    name=my_run

# All text tasks
python -m evaluation.eval_launcher \
    eval=text/text_pipeline \
    model=bert_base \
    name=my_run
```

Override `model=…` to swap encoders, or any individual task YAML in
`evaluation/configs/eval/`.

### Train a CLIP+ model

```bash
torchrun --standalone --nproc-per-node=4 \
    -m apps.baselines.clip.training.train \
    name=babyview_clip mode=triple data=ego4d \
    data.train_dataset.manifest_path=/path/to/manifests/train.json \
    data.train_dataset.image_root=/path/to/frames \
    data.val_dataset.manifest_path=/path/to/manifests/val.json
```

(`data=ego4d` selects the multi-frame-per-utterance loader used for
BabyView, Ego4D, and HowTo. For COCO use `data=coco`.)

### Train EgoBabyLLaVA

```bash
sbatch apps/baselines/llava/scripts/phase1_pretrain.sh
sbatch apps/baselines/llava/scripts/phase2_finetune.sh
```

See [`apps/baselines/llava/README.md`](apps/baselines/llava/README.md)
for the three-phase recipe (GPT-2 from scratch → projector → joint
fine-tune).

## Tests

```bash
pixi run -e dev ci   # ruff check + format + typos + pytest
```

GPU- and integration-marked tests are excluded by default; opt in with
`pytest -m gpu` or `pytest -m integration`.

## Licenses

### Code

The majority of EgoBabyVLM is licensed under [CC-BY-NC 4.0](LICENSE),
however portions of the project are available under separate license
terms:

| Component | Path | License |
|---|---|---|
| DINOv2 | `apps/baselines/dinov2/third_party/dinov2/` | Apache License 2.0 (per-file headers) |
| Perception Encoder | `apps/alignment_scoring/third_party/perception_models/` (PE portion) | [Apache License 2.0](apps/alignment_scoring/third_party/perception_models/LICENSE.PE) |
| Perception-LM | `apps/alignment_scoring/third_party/perception_models/` (PLM portion) | [FAIR Noncommercial Research License](apps/alignment_scoring/third_party/perception_models/LICENSE.PLM) |
| LongTail-Swap | `apps/swapbench/third_party/lt_swap/` | [CC-BY-NC 4.0](apps/swapbench/third_party/lt_swap/LICENSE) |
| LLaVA-derived code | `apps/baselines/llava/` | Apache License 2.0 (per-file headers) |

Please retain all upstream copyright notices and license headers when
reusing files from these directories.

### Data

The Data is released CC-by-NC and is intended for benchmarking purposes
only. Some annotations are outputs of Llama 3.1 and subject to its
license
([Llama 3.1 license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE);
[Llama 4 model card](https://github.com/meta-llama/llama-models/tree/main/models/llama4)).
Third-party content pulled from other locations is subject to its own
licenses, and you may have other legal obligations or restrictions that
govern your use of that content.

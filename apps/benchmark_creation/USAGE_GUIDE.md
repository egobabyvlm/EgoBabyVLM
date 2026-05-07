# Usage Guide

This document covers the benchmark generation pipeline and how to extend it.

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Stage-by-Stage Walkthrough](#stage-by-stage-walkthrough)
- [Configuration](#configuration)
- [Shell Scripts & SLURM](#shell-scripts--slurm)
- [Data Layout](#data-layout)
- [Extending the Benchmark](#extending-the-benchmark)
- [Tools & Models](#tools--models)

---

## Pipeline Overview

The benchmark is built in **5 stages**. Stages 2–4 run in parallel for lexical and grammatical tasks.

```
  vocab_sorted.csv (word frequencies from any caption corpus)
                    |
         Stage 1: Vocabulary Curation
           POS-tag, frequency-bin, filter
                    |  longtail_wordlist.csv
            --------+--------
            |                |
     LEXICAL BRANCH   GRAMMATICAL BRANCH
            |                |
     Stage 2A: Build      Stage 3A: Build
     word lists            sentence pairs
     (WordNet + LLM)       (3-LLM-call pipeline)
            |                |
     Stage 2B: Generate   Stage 3B: Generate
     images (Flux)         images (Flux)
            |                |
            +--------+-------+
                     |
         Stage 4: Post-Filtering
           SigLIP2 / VLM alignment scoring
                     |
         Stage 5: Manifest Generation
           Assemble trials into JSON manifests
```

---

## Running the Full Pipeline

The `run_pipeline.sh` script orchestrates all stages end-to-end. It automatically manages a vLLM server, runs independent stages in parallel, and handles cleanup on exit.

> ⚠️ **Submit with `sbatch`, not `bash`.** All scripts under `scripts/` carry `#SBATCH` directives. `bash <script>` runs them on the **login node** with no GPU allocation; only `sbatch <script>` actually submits a SLURM job. See [Shell Scripts & SLURM](#shell-scripts--slurm) for full details.

```bash
# One-time: create the SLURM log directory (#SBATCH --output writes here).
mkdir -p MachineDevBench_logs

# Full pipeline from a corpus (e.g. COCO)
sbatch apps/benchmark_creation/scripts/run_pipeline.sh \
    --dataset coco --name COCO

# Intersection vocabulary from all four corpora
sbatch apps/benchmark_creation/scripts/run_pipeline.sh \
    --dataset all --name ALL

# Test mode (10 items per task — useful for quick validation)
sbatch apps/benchmark_creation/scripts/run_pipeline.sh \
    --dataset coco --name COCO --test

# Custom output directory
sbatch apps/benchmark_creation/scripts/run_pipeline.sh \
    --output-dir my_benchmark \
    --dataset coco --name COCO

# Skip vocab extraction by providing your own vocab_sorted.csv
sbatch apps/benchmark_creation/scripts/run_pipeline.sh \
    --name COCO --vocab-csv path/to/vocab_sorted.csv
```

Check the queued job:

```bash
squeue -u $USER
tail -f MachineDevBench_logs/slurm-<jobid>-pipeline.out
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `./MachineDevBench` | Output directory (matches `outputs_root` in `configs/paths.yaml`) |
| `--dataset` | *(required unless `--vocab-csv`)* | Source corpus: `coco`, `howto100m`, `ego4d`, `babyview`, `all` (intersection of all four) |
| `--name` | `Dataset` | Label used in filenames and reports |
| `--vocab-csv` | *(auto-computed)* | Path to a pre-computed `vocab_sorted.csv` (skips vocab extraction) |
| `--model` | `google/gemma-4-26B-A4B-it` | LLM model served by vLLM |
| `--img-model` | `FLUX.2-klein-4B` | Diffusion model for image generation |
| `--styles` | `"realistic cartoon"` | Space-separated image styles |
| `--num-gpus` | `4` | GPUs for image generation |
| `--test` | off | Test mode: 10 items per task, debug image sizes |
| `--max-nouns-per-category` | `240` | Nouns sampled per semantic category |
| `--max-adjectives` | `80` | Total adjectives |
| `--items-per-gram-category` | `100` | Sentence pairs per grammatical category |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_PORT` | `8000` | Port for the vLLM server |
| `SKIP_IMAGES` | `0` | Set to `1` to skip image generation (Stages 2B & 3B) |
| `SKIP_FILTERING` | `0` | Set to `1` to skip post-filtering (Stage 4) |
| `FILTER_MODEL` | `facebook/PE-Core-L14-336` | VLM model for grammatical post-filtering |

### Execution Order

The script runs stages in dependency order and parallelizes where possible:

1. **Stage 0** — Vocabulary extraction (skipped if `--vocab-csv` is provided)
2. **Stage 1** — Vocabulary curation
3. **Stages 2A + 3A** — Build lexical word lists and grammatical pairs (parallel, vLLM started)
4. **Stages 2B + 3B** — Generate images (parallel, no vLLM needed)
5. **Stage 4** — Post-filtering (parallel, vLLM restarted for grammatical filtering)
6. **Stage 5** — Manifest generation (parallel)

---

## Stage-by-Stage Walkthrough

### Stage 1: Vocabulary Curation

Filters and enriches the raw vocabulary:
1. **Frequency binning** — Log-scale bins: [0,1), [1,2), [2,4), ..., [512,inf)
2. **POS tagging** — NLTK + WordNet cross-reference (NOUN, VERB, ADJ, ADV, OTHER)
3. **Filtering** — Removes stopwords, proper nouns, names, non-alpha, words without WordNet synsets

```bash
python -m apps.benchmark_creation.pipeline.create_vocabulary \
    --vocab-csv path/to/vocab_sorted.csv \
    --output-dir MachineDevBench --name COCO
```

**Output:** `MachineDevBench/COCO_TIMESTAMP/longtail_wordlist.csv`

### Stage 2A: Build Lexical Word Lists

Creates word sets for nouns and adjectives.

**Nouns:** WordNet semantic categorization (animals, food & drink, body parts, ...) + LLM filtering for child-appropriateness + stratified sampling.

```bash
# With LLM filtering (requires a running vLLM server)
python -m apps.benchmark_creation.pipeline.lexical.build_nouns \
    --vocab-dir MachineDevBench/COCO_TIMESTAMP --name COCO \
    --api-base http://localhost:8000/v1 --model google/gemma-4-26B-A4B-it
```

**Adjectives:** Similar pipeline with LLM-generated contrastive phrases.

```bash
python -m apps.benchmark_creation.pipeline.lexical.build_adjectives \
    --vocab-dir MachineDevBench/COCO_TIMESTAMP --name COCO \
    --api-base http://localhost:8000/v1 --model google/gemma-4-26B-A4B-it
```

**Output:** `Lexical/{Nouns,Adjectives}/word_list.json`

### Stage 2B: Lexical Image Generation

Generates images using a diffusion model (e.g. Flux.2-klein-4B). Supports batched generation and multi-GPU parallelism.

```bash
# Nouns (one image per word)
python -m apps.benchmark_creation.pipeline.lexical.generate_noun_images \
    --data-dir MachineDevBench/COCO_TIMESTAMP \
    --model-id black-forest-labs/FLUX.2-klein-4B \
    --styles realistic cartoon --num-gpus 4

# Adjectives (pos + neg image per word)
python -m apps.benchmark_creation.pipeline.lexical.generate_adj_images \
    --data-dir MachineDevBench/COCO_TIMESTAMP \
    --model-id black-forest-labs/FLUX.2-klein-4B \
    --styles realistic cartoon
```

All image generation scripts are **resumable** — existing images are skipped on re-run.

**Output:** `Lexical/{Nouns,Adjectives}/{style}/`

### Stage 3A: Build Grammatical Benchmark

Generates sentence pairs for 8 grammatical categories via a **3-LLM-call pipeline**:

1. **Word selection** — Picks a word from the vocabulary pool
2. **Pair generation** — LLM generates caption_a / caption_b (or code derives one deterministically)
3. **Validation** — LLM checks grammar, visual representability, and contrastiveness

Categories: `subject_verb`, `subject_adjective`, `negation`, `order_matters`, `prepositions`, `comparatives`, `counting`, `embedded_relative`.

```bash
python -m apps.benchmark_creation.pipeline.grammatical.build_benchmark \
    --vocab-dir MachineDevBench/COCO_TIMESTAMP --name COCO \
    --api-base http://localhost:8000/v1 --model google/gemma-4-26B-A4B-it
```

**Output:** `Grammatical/gram_{category}/sentence_list.json`

### Stage 3B: Grammatical Image Generation

Generates 2 images per trial with **contrastive prompts** (each image prompt includes category-specific guidance to maximize visual distinguishability).

```bash
python -m apps.benchmark_creation.pipeline.grammatical.generate_images \
    --data-dir MachineDevBench/COCO_TIMESTAMP \
    --model-id black-forest-labs/FLUX.2-klein-4B \
    --styles realistic cartoon --num-gpus 4
```

**Output:** `Grammatical/gram_{category}/imgs/{style}/seq_NN/img_0.png, img_1.png`

### Stage 4: Post-Filtering

Scores generated images against their captions and removes poorly aligned trials.

- **Lexical:** SigLIP2 scores image-caption alignment; requires the correct image to score higher than the distractor.
- **Grammatical:** VLM checks depiction quality and caption distinguishability.

```bash
# Lexical: score and filter
python -m apps.benchmark_creation.pipeline.filtering.post_filter_lexical \
    --data-dir MachineDevBench/COCO_TIMESTAMP --write-filtered

# Grammatical: VLM-based depiction + distinction checks
python -m apps.benchmark_creation.pipeline.filtering.post_filter_grammatical \
    --data-dir MachineDevBench/COCO_TIMESTAMP --write-filtered

# Analyze score distributions
python -m apps.benchmark_creation.pipeline.filtering.compute_distributions \
    --data-dir MachineDevBench/COCO_TIMESTAMP
```

**Output (lexical):** `siglip2_scores_{style}.json`, `word_list_filtered_{style}.json`
**Output (grammatical):** `vlm_scores_{style}.json`

### Stage 5: Manifest Generation

Assembles word lists, sentence lists, and image paths into evaluation-ready manifests.

```bash
# Lexical manifests
python -m apps.benchmark_creation.pipeline.manifests.generate_lexical \
    --data-dir MachineDevBench/COCO_TIMESTAMP --tasks nouns adjectives \
    --styles realistic cartoon

# Grammatical manifests
python -m apps.benchmark_creation.pipeline.manifests.generate_grammatical \
    --data-dir MachineDevBench/COCO_TIMESTAMP --styles realistic cartoon
```

**Output:** `manifest_{task}_{style}.json`

---

## Configuration

### `configs/paths.yaml`

Path configuration for the Python package. Override via `$BENCHMARK_CREATION_PATHS` environment variable.

```yaml
outputs_root: ./MachineDevBench

# Dataset manifests (bundled under resources/)
howto100m_manifest: resources/howto100m_manifest.json
ego4d_manifest: resources/ego4d_manifest.json
coco_captions: resources/coco_metaclip_captions.txt
babyview_manifest: resources/babyview_manifest.txt
```

### `configs/styles.yaml`

Image generation style prefixes prepended to prompts before passing to the diffusion model.

```yaml
cartoon:
  prefix: >-
    A simple children's book illustration, clean lines, bright colors,
    white background, no text, no watermark.

realistic:
  prefix: >-
    A clear, realistic photo of the following scene.
    Plain simple background, well-lit, easy to understand.
```

---

## Shell Scripts & SLURM

The `scripts/` directory contains launcher scripts organized by pipeline stage:

```
scripts/
├── 01_Create_Vocabulary/          run_create_vocabulary.sh
├── 02_Create_Lexical/             run_build_nouns.sh
│                                  run_build_adjectives.sh
│                                  run_generate_lexical_nouns_imgs.sh
│                                  run_generate_lexical_adj_imgs.sh
├── 03_Create_Grammatical/         run_create_grammatical.sh
│                                  run_generate_grammatical_images.sh
├── 04_Post_Filtering/             run_post_filter_lexical.sh
│                                  run_post_filter_lexical_hard.sh
│                                  run_post_filter_grammatical.sh
│                                  run_compute_distributions.sh
└── 05_Manifest_Generation/        run_generate_manifests_lexical.sh
                                   run_generate_manifests_grammatical.sh
```

Scripts that need an LLM (build_nouns, build_adjectives, create_grammatical, post_filter_grammatical) automatically launch a vLLM server when `--model` is provided, wait for it to be ready, run the Python module, and shut down the server on exit.

Simple scripts (image generation, filtering, manifests) just pass all arguments through to the Python module.

### Submitting to SLURM

Every launcher script — including `run_pipeline.sh` — starts with `#SBATCH` directives that request the partition, GPUs, memory, and wall-time. **These directives are only honored when the script is submitted via `sbatch`.** Running with `bash` (or executing it directly) ignores them and runs the script in your current shell on the login node, with no GPU allocation. That is almost never what you want.

The shipped scripts intentionally omit `--qos`, `--account`, and `--partition` since they're cluster-specific. Pass them on the `sbatch` command line, or set `SBATCH_*` env vars (e.g. `SBATCH_QOS`, `SBATCH_ACCOUNT`, `SBATCH_PARTITION`) before submitting.

```bash
# ✅ Correct — submits a SLURM job, GPUs are allocated, log files appear
#    under MachineDevBench_logs/slurm-<jobid>-*.{out,err}
sbatch --qos=<your_qos> --account=<your_account> \
    apps/benchmark_creation/scripts/run_pipeline.sh --dataset coco --name COCO

# ❌ Wrong — runs locally on the login node, no GPUs, may hang or OOM
bash  apps/benchmark_creation/scripts/run_pipeline.sh --dataset coco --name COCO
```

The same pattern applies to every per-stage script:

```bash
sbatch --qos=<your_qos> apps/benchmark_creation/scripts/02_Create_Lexical/run_build_nouns.sh \
    --vocab-dir MachineDevBench/COCO_TIMESTAMP --name COCO

sbatch --qos=<your_qos> apps/benchmark_creation/scripts/03_Create_Grammatical/run_create_grammatical.sh \
    --vocab-dir MachineDevBench/COCO_TIMESTAMP --name COCO --model google/gemma-4-26B-A4B-it

sbatch --qos=<your_qos> apps/benchmark_creation/scripts/04_Post_Filtering/run_post_filter_grammatical.sh \
    --data-dir MachineDevBench/COCO_TIMESTAMP --model google/gemma-4-26B-A4B-it
```

### Logs

Each script writes to `MachineDevBench_logs/slurm-<jobid>-<stage>.{out,err}`, relative to your **submission directory** — submit from the repo root.

> If the log directory does not exist, SLURM **silently drops** stdout/stderr. Always run once before your first submission:
>
> ```bash
> mkdir -p MachineDevBench_logs
> ```

### Monitoring & cancelling

```bash
squeue -u $USER                                  # see your queued/running jobs
scontrol show job <jobid>                        # detailed job info
tail -f MachineDevBench_logs/slurm-<jobid>-*.out # live log
scancel <jobid>                                  # cancel a job
```

### Overriding SBATCH defaults at submission time

Flags passed to `sbatch` itself (before the script path) override the in-script `#SBATCH` headers. Useful for shorter runs / smaller allocations:

```bash
# Short test run on 2 GPUs with a 4h wall-clock
sbatch --time=4:00:00 --gpus=2 \
    apps/benchmark_creation/scripts/run_pipeline.sh --test --dataset coco --name COCO

# Specific node
sbatch --nodelist=h200-001 \
    apps/benchmark_creation/scripts/run_pipeline.sh --dataset coco --name COCO
```

### Running interactively (no SLURM)

If you want to debug locally on a machine where you already have GPUs (e.g. a `srun --pty` shell), `bash`-invocation is fine — the `#SBATCH` lines are simply comments to the shell:

```bash
# From inside an interactive GPU allocation:
srun --gpus=4 --time=2:00:00 --pty bash
bash apps/benchmark_creation/scripts/run_pipeline.sh --dataset coco --name COCO --test
```

---

## Data Layout

All generated outputs are stored under a single root directory with named, timestamped subdirectories:

```
MachineDevBench/COCO_TIMESTAMP/
├── longtail_wordlist.csv               # Stage 1
├── frequency_report.txt
│
├── Lexical/
│   ├── Nouns/
│   │   ├── word_list.json              # Stage 2A (categories + words)
│   │   ├── word_list_filtered_realistic.json  # Stage 4
│   │   ├── siglip2_scores_realistic.json      # Stage 4
│   │   ├── manifest_nouns_realistic.json      # Stage 5
│   │   ├── realistic/{category}/{word}.png    # Stage 2B
│   │   └── cartoon/{category}/{word}.png
│   └── Adjectives/
│       └── (same structure with pos.png, neg.png per word)
│
├── Grammatical/
│   ├── gram_subject_verb/
│   │   ├── sentence_list.json          # Stage 3A
│   │   ├── vlm_scores_realistic.json   # Stage 4
│   │   └── imgs/{style}/seq_00/
│   │       ├── metadata.json
│   │       ├── img_0.png               # Image for caption_a
│   │       └── img_1.png               # Image for caption_b
│   ├── gram_negation/
│   ├── gram_order_matters/
│   └── ...
│
└── db_stats.json                       # Dataset statistics
```

---

## Extending the Benchmark

### Adding a new grammatical category

1. **Define the category** in `benchmark_creation/pipeline/grammatical/prompts.py`:
   - Add an entry to `GRAMMATICAL_TEMPLATES` with `pos`, `template`, and `pair_mode`
   - `pair_mode="llm"`: LLM generates both captions
   - `pair_mode="deterministic"`: LLM generates one, code derives the other

2. **Add any special word filters** in `benchmark_creation/pipeline/grammatical/word_filters.py`

3. **Add image prompt rewriters** (if needed) in `benchmark_creation/pipeline/grammatical/rewriters.py`

4. **No other changes needed** — the pipeline in `build_benchmark.py` iterates over `GRAMMATICAL_TEMPLATES` automatically.

### Adding a new lexical task type

1. Create a new builder in `benchmark_creation/pipeline/lexical/` (follow `build_nouns.py` as a template)
2. Create a corresponding image generator (follow `generate_noun_images.py`)
3. Register the task in `benchmark_creation/task_registry.py`
4. Add a manifest generator or extend `generate_lexical.py`

### Adding a new image style

Add an entry to `configs/styles.yaml` with a `prefix` key. All image generation scripts read styles from this file automatically.

### Adding a new corpus

Add the manifest file to `resources/` and register it in `configs/paths.yaml` and `benchmark_creation/paths.py` (add a `get_<name>_manifest()` accessor).

---

## Tools & Models

| Component | Tool |
|-----------|------|
| POS tagging & filtering | NLTK + WordNet |
| Semantic categorization | WordNet hypernym chains |
| Text generation & filtering | LLM via vLLM (e.g. google/gemma-4-26B-A4B-it) |
| Image generation | Flux.2-klein (diffusion model) |
| Lexical post-filtering | Perception Encoder / SigLIP2 / CLIP (image-text alignment scoring) |
| Grammatical post-filtering | VLM (depiction & distinction checks) |
| Job scheduling | SLURM (self-submitting shell scripts) |

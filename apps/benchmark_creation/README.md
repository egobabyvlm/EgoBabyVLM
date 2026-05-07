# Machine-DevBench — Benchmark Creation

A scalable, corpus-grounded pipeline for generating developmental benchmarks that evaluate lexical and grammatical competence in vision-language models. Machine-DevBench draws its vocabulary directly from a model's training corpus, eliminating confounds between vocabulary coverage and linguistic competence. Words are sampled across logarithmic frequency bins covering the full long-tail distribution.

This package (`benchmark_creation/`) contains the **generation pipeline only**. For evaluation, see the `evaluation/` package.


## Tasks

| Task | ID | What it tests |
|------|----|---------------|
| **Nouns** | `lex_nouns` | Word-to-image recognition, same-category distractor |
| **Adjectives** | `lex_adjectives` | Property recognition, antonym contrast |
| **Subject–Verb** | `gram_subject_verb` | Agent–action binding |
| **Subject–Adjective** | `gram_subject_adjective` | Property–object binding |
| **Negation** | `gram_negation` | "is X" vs. "is not X" |
| **Word Order** | `gram_order_matters` | Thematic role assignment (who does what to whom) |
| **Prepositions** | `gram_prepositions` | Spatial relation understanding |
| **Comparatives** | `gram_comparatives` | Comparative constructions |
| **Counting** | `gram_counting` | Numeral comprehension |
| **Embedded Relative** | `gram_embedded_relative` | Relative clause attachment |

All stimuli are generated in two visual styles (photorealistic and cartoon).


## Installation

```bash
pip install -e .                         # Core package only
pip install -e ".[generation]"           # + benchmark generation dependencies
pip install -e ".[generation,dev]"       # + dev tools (ruff, pytest, etc.)
```

## Quick Start

```bash
# Verify installation
python -c "import apps.benchmark_creation; print(benchmark_creation.__version__)"

# Check available tasks
python -c "from apps.benchmark_creation.task_registry import list_tasks; print(list_tasks())"

# One-time: SLURM logs land in MachineDevBench_logs/ (must exist beforehand)
mkdir -p MachineDevBench_logs

# Submit the full pipeline as a SLURM job (use sbatch, NOT bash —
# the #SBATCH directives are only honored by sbatch).
sbatch apps/benchmark_creation/scripts/run_pipeline.sh \
    --dataset coco --name COCO

# Check status / tail logs
squeue -u $USER
tail -f MachineDevBench_logs/slurm-<jobid>-pipeline.out
```

Run a single Python module without SLURM (for local debugging):

```bash
python -m apps.benchmark_creation.pipeline.create_vocabulary \
    --vocab-csv path/to/vocab_sorted.csv --output-dir data/coco --name COCO
```

See [`USAGE_GUIDE.md`](USAGE_GUIDE.md#shell-scripts--slurm) for the full SLURM submission reference (per-stage examples, log management, overriding SBATCH defaults).

Path overrides: set `BENCHMARK_CREATION_PATHS` to a YAML file whose keys override `configs/paths.yaml`.


## Pipeline

The benchmark is built through a 5-stage pipeline, each with a launcher script in `scripts/`:

```
Stage 1  Vocabulary curation     scripts/01_Create_Vocabulary/
Stage 2  Lexical tasks           scripts/02_Create_Lexical/
Stage 3  Grammatical tasks       scripts/03_Create_Grammatical/
Stage 4  Post-filtering          scripts/04_Post_Filtering/
Stage 5  Manifest generation     scripts/05_Manifest_Generation/
```

Configuration lives in `configs/paths.yaml` and `configs/styles.yaml`. See [USAGE_GUIDE.md](USAGE_GUIDE.md) for the full walkthrough.


## Package Structure

```
benchmark_creation/
├── __init__.py
├── paths.py                     # Centralized path config (reads configs/paths.yaml)
├── task_registry.py             # Task definitions (lex_nouns, lex_adjectives, gram)
│
├── configs/
│   ├── paths.yaml               # Dataset & output paths
│   └── styles.yaml              # Image generation style prefixes
│
├── resources/                   # Bundled dataset manifests
│   ├── coco_metaclip_captions.txt   # COCO captions (one per line, MetaCLIP-cleaned)
│   ├── howto100m_manifest.json
│   ├── ego4d_manifest.json
│   └── babyview_manifest.txt
│
├── utils/
│   ├── vocabulary.py            # VocabEntry, POS tagging, frequency binning, CSV I/O
│   ├── flux_pipeline.py         # FluxPipeline wrapper (text-to-image generation)
│   ├── vision_scoring.py        # SigLIP2 / CLIP image-text scoring
│   └── vllm_server.py           # vLLM server launcher (local & SLURM)
│
├── pipeline/
│   ├── vocab_coverage.py        # Optional: vocabulary analysis utility
│   ├── create_vocabulary.py     # Stage 1: POS-tag, filter, bin vocabulary
│   ├── merge_vocabularies.py    # Merge multiple corpus vocabularies
│   ├── db_stats.py              # Compute dataset statistics
│   │
│   ├── lexical/                 # Stage 2: Lexical tasks
│   │   ├── constants.py
│   │   ├── build_nouns.py       # Word lists (WordNet + LLM filtering)
│   │   ├── build_adjectives.py  # Adjective word lists + contrastive phrases
│   │   ├── generate_noun_images.py   # Multi-GPU batched image generation
│   │   └── generate_adj_images.py
│   │
│   ├── grammatical/             # Stage 3: Grammatical tasks
│   │   ├── prompts.py           # LLM prompt templates for 8 categories
│   │   ├── constants.py         # Morphology helpers, noun allowlists
│   │   ├── diversity.py         # Word/pair reuse caps
│   │   ├── parsers.py           # Response parsing
│   │   ├── rewriters.py         # Prompt rewriters for image generation
│   │   ├── word_filters.py      # Blocked word lists
│   │   ├── build_benchmark.py   # 3-LLM-call pipeline
│   │   └── generate_images.py   # Contrastive image generation
│   │
│   ├── filtering/               # Stage 4: Post-filtering
│   │   ├── compute_distributions.py
│   │   ├── post_filter_lexical.py
│   │   ├── post_filter_lexical_hard.py
│   │   └── post_filter_grammatical.py
│   │
│   └── manifests/               # Stage 5: Manifest generation
│       ├── generate_lexical.py
│       └── generate_grammatical.py
│
└── scripts/                     # Launcher scripts
    ├── 01_Create_Vocabulary/
    ├── 02_Create_Lexical/
    ├── 03_Create_Grammatical/
    ├── 04_Post_Filtering/
    └── 05_Manifest_Generation/
```

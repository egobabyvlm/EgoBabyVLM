# SwapBench: minimal-pair language and visual-property probes

Two corpus-grounded benchmark generators for evaluating language and
visual-property knowledge from minimal pairs:

| Benchmark | Origin | What it tests |
|---|---|---|
| **LT-Swap** (WordSwap, InflectionSwap, AgreementSwap) | Built around [`facebookresearch/lt-swap`](https://github.com/facebookresearch/lt-swap) | Lexical and morphological knowledge of long-tail words sampled from the model's training corpus (see [Algayres et al.](https://arxiv.org/abs/2502.10075)). |
| **VP-Swap** (4 properties: color, material, relative size, shape) | New, paper App. methods:vp-swap | Grounded knowledge of physical-object properties from the same long-tail vocabulary. |

Both pipelines drive an OpenAI-compatible LLM endpoint (a local
[vLLM](https://github.com/vllm-project/vllm) server is the canonical
choice) via the same async worker pool
(`apps/swapbench/utils/llm_runner.py`), so the model behind each step is
swappable from the CLI.

## Layout

```
apps/swapbench/
├── third_party/
│   └── lt_swap/                  # Upstream LT-Swap (CC-BY-NC; see VERSION + LICENSE)
│       ├── generate_task/        # Per-stage prep + filter scripts (kept verbatim from upstream)
│       ├── eval/                 # Upstream evaluator scripts (not on the OSS critical path)
│       ├── LICENSE
│       ├── README.upstream.md    # Upstream README, kept for reference
│       └── VERSION               # Upstream URL + pinned commit SHA
├── longtail_swap/
│   ├── build_word_lists.py       # Stage-0 corpus → wordlist + inflpairs + visualwords
│   └── generate.py               # End-to-end runner (task=wordswap or task=syntax)
├── visual_property_swap/
│   ├── prompts.py                # Per-property prompt templates
│   └── generate.py               # End-to-end VP-Swap runner (one or all four properties)
└── utils/
    └── llm_runner.py             # Async worker pool that drives the LT-Swap pipeline against a vLLM endpoint
```

The upstream code under `third_party/lt_swap/generate_task/` and
`third_party/lt_swap/eval/` is kept byte-identical to the source repo at
the SHA pinned in `VERSION`; refresh is a re-copy + bump. The Hydra
runners under `longtail_swap/` and the VP-Swap pipeline under
`visual_property_swap/` are first-party, drive the upstream scripts as
subprocesses, and use the async worker pool in `utils/llm_runner.py`
instead of the upstream `mp_main.py` orchestrator.

## CLI entry points

```text
egobabyvlm-swapbench-build-word-lists   # corpus → wordlist + inflpairs + visualwords
egobabyvlm-swapbench-lt-swap            # WordSwap or syntax (InflectionSwap + AgreementSwap)
egobabyvlm-swapbench-vp-swap            # VP-Swap, one property or all four
```

## Bundled data

This package ships **no pair files**. Each new training corpus must
regenerate the swap files via the pipelines below. The reason is
pragmatic — pair files are model-conditional (which LLM produced them
matters for the resulting bias), and the LLM-call stage is by far the
most expensive part of the pipeline, so caching them in-tree would
encourage stale artifacts.

## Quickstart

The pipelines assume an OpenAI-compatible inference endpoint. Start a
vLLM server with your judge model first; the example below uses
[Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct),
the model used in the EgoBabyVLM paper.

```bash
# Start a vLLM server (reuses apps/benchmark_creation/utils/vllm_server.py).
python -m apps.benchmark_creation.utils.vllm_server local \
    --model meta-llama/Llama-3.1-405B-Instruct \
    --port 8000
```

### 1. Build the per-corpus word lists (one-time per corpus)

```bash
egobabyvlm-swapbench-build-word-lists \
    processor.data_dir=/path/to/corpus_text/ \
    processor.output_dir=/path/to/wordlists/
```

The corpus directory should contain plain `.txt` files (one shard per
file; the upstream `get_word_lists.py` parallelises over shards). Output:

```
/path/to/wordlists/
├── wordlists/             # per-shard intermediate JSONs
├── longtail_wordlist      # WordSwap candidate words
├── longtail_inflpairs     # InflectionSwap / AgreementSwap candidate pairs
├── longtail_visualnouns   # VP-Swap candidate words (word,freq per row, nouns only)
└── vocabulary             # corpus vocabulary with raw frequency counts
```

### 2. Run LT-Swap

```bash
# WordSwap
egobabyvlm-swapbench-lt-swap \
    processor.task=wordswap \
    processor.wordlists_dir=/path/to/wordlists/ \
    processor.output_dir=/path/to/swapbench/wordswap/ \
    processor.model=meta-llama/Llama-3.1-405B-Instruct

# InflectionSwap + AgreementSwap (shared pipeline; produces both files)
egobabyvlm-swapbench-lt-swap \
    processor.task=syntax \
    processor.wordlists_dir=/path/to/wordlists/ \
    processor.output_dir=/path/to/swapbench/syntax/ \
    processor.model=meta-llama/Llama-3.1-405B-Instruct
```

Each stage writes to disk and is restartable — re-running the same
command picks up where the last one left off. The stages are documented
in the `apps/swapbench/longtail_swap/generate.py` module docstring.

### 3. Run VP-Swap

```bash
# All four properties sequentially
egobabyvlm-swapbench-vp-swap \
    processor.visualnouns_path=/path/to/wordlists/longtail_visualnouns \
    processor.output_dir=/path/to/swapbench/vp_swap/ \
    processor.visual_property=all \
    processor.model=meta-llama/Llama-3.1-405B-Instruct

# Single property
egobabyvlm-swapbench-vp-swap \
    processor.visualnouns_path=/path/to/wordlists/longtail_visualnouns \
    processor.output_dir=/path/to/swapbench/vp_swap/ \
    processor.visual_property=color \
    processor.model=meta-llama/Llama-3.1-405B-Instruct
```

The first stage (the "is this word physical?" gate) is shared across
properties and only runs once.

## Output schema

LT-Swap pair files match the upstream format documented in
`apps/swapbench/third_party/lt_swap/README.upstream.md`. VP-Swap pair
files use the LT-Swap `visualswap` row layout
(`bin|VISUAL|w1|s1|i1|w2|s2|i2`), one file per property.

To evaluate a model on these pair files, point the LT-Swap evaluator
(already shipped at `evaluation/text/ltswap.py`) at the pair files
produced above.

## License

The upstream LT-Swap code under `third_party/lt_swap/` is CC-BY-NC and
retains all [upstream copyright headers](third_party/lt_swap/LICENSE).
Code under `apps/swapbench/longtail_swap/`,
`apps/swapbench/visual_property_swap/`, and `apps/swapbench/utils/` is
part of EgoBabyVLM and inherits the top-level [CC-BY-NC](../../LICENSE)
license.

Generated pair files are derivative works of the LLM that produced them
and are subject to that LLM's license. For Llama-class models, see the
[Llama 3.1 license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE).

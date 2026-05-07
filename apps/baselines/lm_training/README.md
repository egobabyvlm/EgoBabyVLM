# Language-model pretraining baselines

Trains language models from scratch on a plain-text corpus (one utterance per
line). The trained checkpoints feed downstream stacks: GPT-2 backs the LLaVA
multimodal baseline at `apps/baselines/llava/`, BERT backs the contrastive
trainer's `interleaved_lm` mode at `apps/baselines/clip/training/`.

| Model | Trainer | Tokenizer training |
|-------|---------|--------------------|
| GPT-2 | [`train/train_gpt2.py`](train/train_gpt2.py) | bundled (retrains BPE from `gpt2`) |
| BERT  | [`train/train_bert.py`](train/train_bert.py) | [`scripts/train_bert_tokenizer.py`](scripts/train_bert_tokenizer.py) |

## GPT-2 from scratch

The GPT-2 trainer also retrains a byte-level BPE tokenizer from the standard
GPT-2 base on your training corpus (via HuggingFace's `train_new_from_iterator`).
Pass `--tokenizer_name <hf_id_or_path>` to use an existing tokenizer instead.

### SLURM

```bash
EGOBABYVLM_DATA_DIR=/path/to/your/data \
EGOBABYVLM_CKPT_DIR=/path/to/your/checkpoints \
sbatch --qos=<your_qos> --account=<your_account> \
    apps/baselines/lm_training/scripts/phase0_train_gpt2.sh
```

Tunables (env vars): `PHASE0_FORMAT`, `TOKENIZER_MODE` (`custom` / `mistral`),
`SEED`, `LR`, `BS`, `GACC`, `EPOCHS`. See the script header.

### Direct

```bash
python -m apps.baselines.lm_training.train.train_gpt2 \
    --train_file /path/to/corpus_train.txt \
    --validation_file /path/to/corpus_val.txt \
    --output_dir /path/to/output \
    --vocab_size 52000 \
    --do_train --do_eval \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 30 \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch
```

## BERT from scratch (MLM)

Three steps, each with its own script:

### 1. Train a WordPiece tokenizer on your corpus

```bash
python -m apps.baselines.lm_training.scripts.train_bert_tokenizer \
    /path/to/output/tokenizers/bert_corpus \
    --train_file /path/to/corpus_train.txt \
    --val_file   /path/to/corpus_val.txt \
    --vocab_size 30522
```

The tokenizer inherits algorithm + special-token layout from `bert-base-cased`
by default; override with `--base_tokenizer <hf_id>`.

### 2. Build a fresh BERT config

```bash
python -m apps.baselines.lm_training.scripts.create_bert_config \
    /path/to/output/configs/bert_base
```

Emits a `BertConfig` matching `bert-base-cased` (12 layers, 768 hidden,
30522 vocab). Override architecture knobs with `--hidden_size`,
`--num_hidden_layers`, `--num_attention_heads`, `--intermediate_size`,
`--max_position_embeddings`, etc.

### 3. Run the MLM trainer

```bash
TRAIN_FILE=/path/to/corpus_train.txt \
VAL_FILE=/path/to/corpus_val.txt \
TOKENIZER_FOLDER=/path/to/output/tokenizers/bert_corpus \
CONFIG_FOLDER=/path/to/output/configs/bert_base \
EGOBABYVLM_CKPT_DIR=/path/to/output \
sbatch --qos=<your_qos> --account=<your_account> \
    apps/baselines/lm_training/scripts/train_bert.sh
```

Tunables (env vars): `MODEL_DIR`, `LR`, `NUM_TRAIN_EPOCHS`, `PER_GPU_BATCH_SIZE`,
`MLM_PROBABILITY`, `SEED`, `EGOBABYVLM_LOG_DIR`. See the script header.

The trained BERT checkpoint plugs into the contrastive trainer's
`text_encoder.hf_model_name` config — point that at the output dir.

## Output layout

```
<EGOBABYVLM_CKPT_DIR>/
├── phase0_gpt2/gpt2_<tok_tag>_<format_tag>/  # GPT-2 trainer
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer/                            # retrained BPE
│   └── ...
└── bert_mlm/                                 # BERT trainer
    ├── config.json
    ├── pytorch_model.bin
    └── ...
```

GPT-2 output → LLaVA Phase 1 / Phase 2's `GPT2_MODEL` env var.
BERT output → the contrastive trainer's `text_encoder.hf_model_name` config.

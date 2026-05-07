# Alignment scoring

`apps/alignment_scoring/` provides four pipelines for measuring how well
images or videos align with their captions, plus tooling to re-caption a
dataset and shuffle manifests for negative-pair construction.

## Pipelines

| Pipeline | Question it answers |
|---|---|
| **CLIP scoring** | Are matched (image, caption) cosine similarities meaningfully higher than shuffled? |
| **STS scoring** | When we paraphrase a caption manifest, do the new captions stay close to the originals (vs random other captions)? |
| **Captioning** | Re-caption a dataset with Perception-LM. |
| **VQA scoring** | When we ask Perception-LM "Does this image show 'X'?", is P(Yes) higher for matched vs shuffled captions? |

All three scoring pipelines schedule a **matched** + **shuffled** processor in
parallel via Stopes and aggregate the two cosine-sim / score distributions
into bootstrap JS divergence + KL stats. The output `results.yaml` includes
the divergence summary plus per-pair CSVs for downstream analysis.

## Manifest formats

Two formats are supported, distinguished by extension:

- `*.json` — COCO-format with top-level `images` and `annotations` arrays.
- `*.csv` — must include `clip_filename` and `utterance` columns; extra columns
  are ignored.

For the matched-vs-shuffled scoring pipelines you need two manifests over the
same media: a "matched" one with the original captions and a "shuffled" one
where captions are randomly reassigned to other media. Use the bundled
`alignment-create-shuffled-manifest` CLI to produce the shuffled side
deterministically:

```bash
alignment-create-shuffled-manifest \
  --manifest-path captions_train2017.json \
  --output-path captions_train2017_shuffled.json \
  --type json --random-seed 42
```

## Datasets supported

COCO and Ego4D ship as concrete instantiations. Adding a new dataset is one
Python class + one YAML — see `apps/alignment_scoring/README.md`.

## Models

| Model YAML | Used by | Notes |
|---|---|---|
| `vit_b16_openai` | CLIP scoring | Off-the-shelf OpenAI CLIP ViT-B/16; smallest model, useful for smoke tests |
| `pe_core_bigg` | CLIP scoring (default) | Perception Encoder Core bigG-14 at 448px; what the paper used |
| `plm_1b` | Captioning, VQA | Perception-LM 1B; smallest PLM variant |
| `plm_8b` | Captioning, VQA (default) | Perception-LM 8B; what the paper used |
| `sonar_text` | STS scoring | SONAR text encoder (basic) |

## Output structure

Each pipeline writes to `output_dir`. Common outputs:

```
<output_dir>/
├── results.yaml                       # JSD + KL + per-side mean/std
├── js_bootstrap_distribution.npy      # bootstrap JS samples
├── similarity_histogram.png           # KDE plot (matched vs shuffled)
├── cosine_similarities.csv            # CLIP scoring only
├── sts_results_{matched,shuffled}.csv # STS scoring only
├── vqa_results_{matched,shuffled}.csv # VQA scoring only
└── recaptioned.json                   # Captioning only
```

## SLURM

Every pipeline defaults to local execution. For SLURM submission:

```bash
alignment-clip-scoring ... \
  launcher.cluster=slurm \
  launcher.update_parameters.slurm_qos=high \
  launcher.update_parameters.slurm_account=my-account
```

Stopes job arrays auto-shard the dataset by `num_items_per_chunk` (default
2000). Tune `cpus_per_task` and `gpus_per_node` via the launcher overrides
if you need to.

## Limitations / known gaps

- **PLM smoke tests are not gated yet.** The captioning / VQA pipelines have
  no GPU smoke test because PLM-1B downloads >3GB and PLM-8B >16GB on first
  run. Tracked as follow-up.

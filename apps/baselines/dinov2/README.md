# DINOv2 baseline

Self-supervised vision pretraining (DINO + iBOT objectives) and feature
extraction. Code forked from [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
and adapted for OSS distribution.

## Layout

```
apps/baselines/dinov2/
├── extractor.py                    # ImageFeatureExtractor wrapper for the eval pipeline
├── training/
│   ├── train.py                    # SSL training entrypoint
│   └── submit.py                   # Submitit-based SLURM submission entrypoint
├── scripts/
│   └── train_dinov2.sh             # SLURM launcher
└── third_party/dinov2/             # Upstream dinov2 library, kept under its own README
```

`third_party/dinov2/` carries its own README documenting the upstream
provenance and the local refresh procedure.

## Training

### Configs

The shipped configs under `third_party/dinov2/configs/train/` cover several
setups (BabyView, Ego4D, HowTo100M, ImageNet). Each is a Hydra-style YAML
referencing a dataset by the `dataset_path` string, e.g.::

    dataset_path: ImageNet;split=TRAIN;root=${oc.env:IMAGENET_ROOT};extra=${oc.env:IMAGENET_EXTRA}

Available dataset names (from `third_party/dinov2/data/datasets/__init__.py`):
`ImageNet`, `MSCOCO`, `Ego4D`, `HowToSubset` (referenced as `HowTo`),
`BabyView`. All operate on local-FS paths only; provide them via env vars
referenced from the YAML or by editing the config.

### Submitting via SLURM

```bash
CONFIG_FILE=apps/baselines/dinov2/third_party/dinov2/configs/train/vitb14_imagenet.yaml \
EGOBABYVLM_CKPT_DIR=/path/to/checkpoints \
sbatch --qos=<your_qos> --account=<your_account> \
    apps/baselines/dinov2/scripts/train_dinov2.sh
```

Tunables (env vars): `OUTPUT_DIR`, `OPTS` (Hydra overrides). See the script
header for details.

### Running directly

```bash
python -m apps.baselines.dinov2.training.train \
    --config-file apps/baselines/dinov2/third_party/dinov2/configs/train/vitb14_imagenet.yaml \
    train.output_dir=/path/to/output \
    train.batch_size_per_gpu=64
```

### Submitting via the bundled Submitit driver

```bash
python -m apps.baselines.dinov2.training.submit \
    --config-file apps/baselines/dinov2/third_party/dinov2/configs/train/vitb14_imagenet.yaml \
    --partition <slurm_partition> \
    --output-dir /path/to/output \
    --ngpus 8
```

This is the upstream DINOv2 entrypoint; prefer the SLURM script above
for new runs since it threads our env-var conventions cleanly.

## Output layout

```
<OUTPUT_DIR>/
├── config.yaml                    # the resolved DINOv2 config used for the run
├── eval/
│   └── training_<step>/
│       └── teacher_checkpoint.pth # periodic teacher snapshots
├── model_<step>.rank_<r>.pth      # FSDP-sharded student/teacher
└── logs/
    └── log.txt
```

The `teacher_checkpoint.pth` files plug directly into the
`DINOv2FeatureExtractor` at `apps/baselines/dinov2/extractor.py`.

## Evaluating a trained checkpoint

```bash
python -m evaluation.eval_launcher \
    eval=vision/knn_imagenet \
    model=dino \
    +model.kwargs.pretrained_weights=/path/to/teacher_checkpoint.pth \
    +model.kwargs.config_file=/path/to/output/config.yaml \
    eval.output_dir=$HOME/dinov2_eval
```

## License

Apache License 2.0 (inherited from facebookresearch/dinov2; see file headers).

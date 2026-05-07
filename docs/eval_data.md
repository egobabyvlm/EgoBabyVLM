# Evaluation datasets

The evaluation pipeline resolves dataset paths at runtime via Hydra environment
variables. To run any eval, download the data once (using the scripts in
`scripts/eval_data/`), point the corresponding env var at your cache, and Hydra
will substitute it into the YAML.

## Cache layout

Default cache root is `~/.cache/egobabyvlm/eval_data/`. Override with
`EGOBABYVLM_CACHE` if you want it elsewhere.

```
~/.cache/egobabyvlm/eval_data/
├── mnist/
├── countbench/
├── devbench/
│   ├── sem-things/
│   ├── gram-trog/
│   ├── ...
├── zorro/
└── cocostuff/
    ├── train2017/
    ├── val2017/
    └── annotations/
```

## Automatable downloads

Each script accepts `--cache-dir <path>` and `--force` (re-download even if
cached). All write a `.downloaded` marker on completion so the next run is a
no-op.

| Dataset | Command | Env var the YAMLs read |
|---|---|---|
| MNIST | `pixi run -e dev python -m scripts.eval_data.download_mnist` | `MNIST_ROOT` |
| CountBench | `pixi run -e dev python -m scripts.eval_data.download_countbench` | `COUNTBENCH_ROOT` |
| Zorro | `pixi run -e dev python -m scripts.eval_data.download_zorro` | `ZORRO_DATA_ROOT` |
| DevBench | `pixi run -e dev python -m scripts.eval_data.download_devbench` | `DEVBENCH_DATA_ROOT` |
| COCO-Stuff (~20 GB) | `pixi run -e dev python -m scripts.eval_data.download_cocostuff` | `COCOSTUFF_ROOT` |

Or grab everything at once:

```sh
pixi run -e dev python -m scripts.eval_data.download_all
```

### Notes

- **DevBench**: the `gram-winoground` task is gated on HuggingFace
  (`facebook/winoground`). Run `huggingface-cli login` once and accept the
  dataset's terms before running the download. The other five DevBench tasks
  are open-access.
- **COCO-Stuff** is large; the script streams to a temporary `_downloads/` dir
  inside your cache and removes it after extraction.
- **Zorro** is downloaded as raw paradigm `.txt` files from the
  [phueb/Zorro](https://github.com/phueb/Zorro) repo and converted to per-task
  BLiMP-style JSONLs the pipeline expects.

## Manual setup required

These datasets either have access restrictions, non-trivial preprocessing, or
both. Each YAML that depends on one references an env var the operator must set
to point at a pre-prepared dataset directory.

### ImageNet (`IMAGENET_ROOT`)

ImageNet is gated on [HuggingFace](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
and needs preprocessing into the directory layout DINOv2's
`make_dataset(dataset_str=...)` expects:

```
<IMAGENET_ROOT>/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_18.JPEG
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

Plus a separate `extra/` directory for DINOv2's class-id metadata. Refer to the
[DINOv2 ImageNet preparation guide](https://github.com/facebookresearch/dinov2/blob/main/README.md#data-preparation)
for the exact `extra/` artifacts required.

After preparing, set:

```sh
export IMAGENET_ROOT=/path/to/imagenet_fullsize
export IMAGENET_EXTRA=/path/to/imagenet_extra
```

Then KNN/Linear/ABX ImageNet evals will work.

### NYUv2 depth (`NYU_ROOT`)

The depth-estimation eval expects the standard NYUv2 monocular-depth split
(introduced by BTS, Lee et al. 2019, and reused by AdaBins, NeWCRFs, ZoeDepth,
and most NYUv2 monocular-depth papers since), not the raw
`nyu_depth_v2_labeled.mat` file. Layout:

```
<NYU_ROOT>/
├── nyu_train.txt          # one line per sample: rgb_path depth_path focal_length
├── nyu_test.txt
└── <scene>/
    ├── rgb_<N>.jpg
    └── sync_depth_<N>.png # 16-bit depth in millimeters
```

Source: download the raw NYUv2 RGBD sync from
[the NYU lab](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip)
and follow the [BTS data prep](https://github.com/cleinc/bts/tree/master/utils)
(both `extract_official_train_test_set_from_mat.py` for the test split and
`sync_project_frames_multi_threads.m` for the train split) to produce the
RGB/depth pairs in the expected layout. BTS ships the split files at
`train_test_inputs/nyudepthv2_{train,test}_files_with_gt.txt` — symlink or copy
those to `<NYU_ROOT>/nyu_{train,test}.txt` (the line format
`rgb_path depth_path focal_length` is already what our loader expects).

### LT-Swap (`LTSWAP_DATA_ROOT`)

> **TODO**: document the LT-Swap data preparation steps. The eval expects
> `LTSWAP_DATA_ROOT` to contain four sentence-pair files generated from the
> training dataset's vocabulary:
>
> - `wordswap_sentence_pairs_filtering_prompts_final_output.txt`
> - `visual_sentence_filtering_prompts_final_output.txt`
> - `agrswap_sentence_pairs`
> - `inflswap_sentence_pairs`
>
> Generation tooling will be added when the LM training pipeline is ported.

## Running an eval against your cache

After exporting the relevant env vars, the eval YAMLs Just Work:

```sh
export MNIST_ROOT=~/.cache/egobabyvlm/eval_data/mnist
pixi run -e dev python evaluation/eval_launcher.py \
    eval=vision/knn_mnist \
    model=dino \
    eval.output_dir=/tmp/knn_mnist_run \
    launcher.cluster=local
```

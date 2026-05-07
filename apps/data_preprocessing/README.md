# Data preprocessing

Four pipelines for turning raw video data into the contrastive training
manifests consumed by `apps/baselines/clip/`:

| Pipeline                       | Entry point               | What it does                                                                                  |
| ------------------------------ | ------------------------- | --------------------------------------------------------------------------------------------- |
| **Frame extraction**           | `egobabyvlm-extract-frames`  | Sample frames from video files at a fixed FPS (ffmpeg).                                       |
| **WhisperX transcription**     | `egobabyvlm-transcribe-whisperx` | Word-level audio transcription with [WhisperX](https://github.com/m-bain/whisperX), one JSON per video. |
| **VTC + word-confidence filter** | `egobabyvlm-filter-vtc`     | Drop transcript segments overlapping the key child (KCHI) and/or below a mean word-confidence threshold. |
| **Manifest builder**           | `egobabyvlm-build-clip-manifest` | Pair WhisperX-format transcripts with extracted frames and emit train/val/test JSONs in the trainer schema. |

Each pipeline is a single Hydra entry point. Frame extraction and WhisperX
fan out via [`stopes`](https://github.com/facebookresearch/stopes) job
arrays — set `launcher.cluster=slurm` to dispatch on a cluster, or leave
the default (`local`) for a single-host run. The VTC filter is CPU-bound
JSON parsing and runs as `joblib.Parallel` within a single process. The
manifest builder is single-process and finishes in seconds even on
~50k-video corpora.

## Layout

```
apps/data_preprocessing/
├── frames/
│   └── extract_frames.py            # ffmpeg frame extraction (stopes job array)
├── transcription/
│   ├── whisperx_transcribe.py       # WhisperX inference (stopes job array, GPU)
│   └── filter_with_vtc.py           # KCHI + word-confidence filter (joblib)
└── manifests/
    └── build_manifest.py            # Pair transcripts + frames → train/val/test JSON
```

## Quickstart

### 1. Extract frames

```bash
egobabyvlm-extract-frames \
    processor.data_dir=/path/to/videos \
    processor.output_dir=/path/to/output \
    processor.fps=1 \
    processor.videos_per_chunk=100
```

Frames land at `<output_dir>/frames/<video_name>/<video_name>_<idx>.jpg`,
plus a per-run summary JSON at the output root. Override
`processor.video_extensions` to include other container formats. The
1-indexed `<idx>` is what the manifest builder expects (see step 4).

### 2. Transcribe with WhisperX

```bash
egobabyvlm-transcribe-whisperx \
    processor.data_dir=/path/to/videos \
    processor.output_dir=/path/to/output \
    processor.whisperx_model=large-v2 \
    processor.batch_size=16 \
    processor.language=en
```

Per-video transcripts land at `<output_dir>/transcriptions/<video_name>.json`
with WhisperX's standard `segments` + `words` schema. A
`transcription_summary.json` is written at the output root summarising success
counts, per-language distribution, and total word counts.

### 3. Filter transcripts (VTC + word confidence)

```bash
egobabyvlm-filter-vtc \
    processor.transcripts_dir=/path/to/whisperx_output/transcriptions \
    processor.vtc_annotations_dir=/path/to/vtc_rttms \
    processor.output_dir=/path/to/filtered \
    processor.num_workers=8 \
    processor.min_avg_word_score=0.5
```

Transcripts and RTTM files are matched by filename stem. For each pair the
filter:

1. drops every segment whose interval overlaps a `KCHI` (key child)
   annotation, then
2. (if `min_avg_word_score` is set) drops every remaining segment whose
   mean WhisperX word-confidence is below the threshold.

The filtered transcript is written to `<output_dir>/<original_filename>.json`
with extra metadata fields (`kchi_filtered`, `kchi_segments_removed`,
`low_confidence_segments_removed`, `min_avg_word_score`). A
`filter_summary.json` aggregates per-VTC-label confidence statistics across
the full input set. **VTC filtering is BabyView-specific** (key-child speech
removal); for HowTo100M / Ego4D this step is skipped.

### 4. Build the train/val/test manifest

```bash
egobabyvlm-build-clip-manifest \
    processor.transcripts_dir=/path/to/transcripts \
    processor.frames_dir=/path/to/output/frames \
    processor.output_dir=/path/to/manifests \
    processor.frames_fps=1 \
    processor.train_frac=0.85 \
    processor.val_frac=0.10 \
    processor.min_frames_per_utterance=1 \
    processor.seed=42
```

Pairs each transcript JSON (WhisperX or VTC-filtered output, same schema)
with the corresponding `<frames_dir>/<video_name>/` directory and writes
three flat JSON lists — `train.json`, `val.json`, `test.json` — plus a
`manifest_build_summary.json`. The frame index → time mapping uses the
midpoint convention `t = (idx − 0.5) / frames_fps`, so make sure
`frames_fps` matches the `processor.fps` you passed to
`egobabyvlm-extract-frames`.

The output schema is exactly what
`apps.baselines.clip.data.HowToCaptionsDataset` and
`Ego4DCaptionsDataset` consume:

```json
[
  {
    "utterance": "the cat sat on the mat",
    "frame_filenames": ["vid_a/vid_a_3.jpg", "vid_a/vid_a_4.jpg"],
    "timestamps": [2.5, 3.5],
    "utterance_num": 1,
    "video_filename": "vid_a.mp4",
    "transcript_filename": "vid_a.json",
    "num_frames": 2
  }
]
```

So the trainer override is just:

```bash
data.train_dataset.manifest_path=/path/to/manifests/train.json
data.train_dataset.image_root=/path/to/output/frames
data.val_dataset.manifest_path=/path/to/manifests/val.json
```

## Submitting on SLURM

Both stopes-driven pipelines (`egobabyvlm-extract-frames`,
`egobabyvlm-transcribe-whisperx`) accept Hydra launcher overrides:

```bash
egobabyvlm-transcribe-whisperx \
    launcher.cluster=slurm \
    launcher.update_parameters.slurm_qos=<your_qos> \
    launcher.update_parameters.slurm_account=<your_account> \
    processor.data_dir=/path/to/videos \
    processor.output_dir=/path/to/output
```

The job array slices `processor.videos_per_chunk` videos per task; tune
that and the per-task `Requirements` (in `extract_frames.py` /
`whisperx_transcribe.py`) for your cluster.

## Per-dataset recipes

### BabyView

[BabyView](https://databrary.org/volumes/1882) is a longitudinal corpus
of head-mounted-camera footage from young children.

1. **Download BabyView** from
   [Databrary volume 1882](https://www.databrary.org/volume/1882). You
   will need a Databrary account and the appropriate access agreement.
2. **Run VTC speaker diarization** on each video with
   [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC). Save the resulting
   RTTM files to a directory mirroring the video filenames. The `KCHI`
   (key child) label marks segments where the target child speaks; we
   filter those out since the child is on-camera and their utterances
   are not paired with the visual scene.
3. **Transcribe with WhisperX**:
   ```bash
   egobabyvlm-transcribe-whisperx \
       processor.data_dir=/path/to/babyview/videos \
       processor.output_dir=/path/to/babyview/whisperx \
       processor.whisperx_model=large-v2 \
       processor.language=en
   ```
4. **Filter** with both KCHI removal and a word-confidence floor (we used
   `min_avg_word_score=0.5` in the paper; see the per-label statistics
   in your `filter_summary.json` to pick a threshold for your data):
   ```bash
   egobabyvlm-filter-vtc \
       processor.transcripts_dir=/path/to/babyview/whisperx/transcriptions \
       processor.vtc_annotations_dir=/path/to/babyview/vtc_rttms \
       processor.output_dir=/path/to/babyview/filtered_transcripts \
       processor.min_avg_word_score=0.5
   ```
5. **Extract frames** at 1 FPS from the source videos:
   ```bash
   egobabyvlm-extract-frames \
       processor.data_dir=/path/to/babyview/videos \
       processor.output_dir=/path/to/babyview/frames_output \
       processor.fps=1
   ```
6. **Build the manifest**:
   ```bash
   egobabyvlm-build-clip-manifest \
       processor.transcripts_dir=/path/to/babyview/filtered_transcripts \
       processor.frames_dir=/path/to/babyview/frames_output/frames \
       processor.output_dir=/path/to/babyview/manifests \
       processor.frames_fps=1
   ```
7. **Train**, pointing the trainer at the manifest + frames root.
   BabyView and Ego4D share the same manifest schema (one utterance per
   record, multiple frames per utterance), so both use the `data=ego4d`
   config:
   ```bash
   torchrun --standalone --nproc-per-node=4 \
       -m apps.baselines.clip.training.train \
       data=ego4d \
       data.train_dataset.manifest_path=/path/to/babyview/manifests/train.json \
       data.train_dataset.image_root=/path/to/babyview/frames_output/frames \
       data.val_dataset.manifest_path=/path/to/babyview/manifests/val.json
   ```

### Ego4D

[Ego4D](https://ego4d-data.org/) is a large egocentric video dataset.

1. **Request access** at https://ego4d-data.org/ (sign the data use
   agreement) and install the official downloader.
2. **Download the full-scale videos** with the upstream CLI; pick the
   subset relevant to your study:
   ```bash
   pip install ego4d
   ego4d --output_directory=/path/to/ego4d --datasets full_scale
   ```
   (Ego4D ships its own narration JSONs under `--datasets annotations`,
   but we don't use those — we re-transcribe with WhisperX so the
   transcript schema and word-level timestamps match the rest of the
   stack.)
3. **Transcribe with WhisperX** (no VTC step — Ego4D videos are not
   centered on a target child):
   ```bash
   egobabyvlm-transcribe-whisperx \
       processor.data_dir=/path/to/ego4d/v2/full_scale \
       processor.output_dir=/path/to/ego4d/whisperx \
       processor.whisperx_model=large-v2 \
       processor.language=en
   ```
4. **Extract frames** at 1 FPS:
   ```bash
   egobabyvlm-extract-frames \
       processor.data_dir=/path/to/ego4d/v2/full_scale \
       processor.output_dir=/path/to/ego4d/frames_output \
       processor.fps=1
   ```
5. **Build the manifest**:
   ```bash
   egobabyvlm-build-clip-manifest \
       processor.transcripts_dir=/path/to/ego4d/whisperx/transcriptions \
       processor.frames_dir=/path/to/ego4d/frames_output/frames \
       processor.output_dir=/path/to/ego4d/manifests \
       processor.frames_fps=1
   ```
6. **Train** (the Ego4D loader is the same as for BabyView):
   ```bash
   torchrun --standalone --nproc-per-node=4 \
       -m apps.baselines.clip.training.train \
       data=ego4d \
       data.train_dataset.manifest_path=/path/to/ego4d/manifests/train.json \
       data.train_dataset.image_root=/path/to/ego4d/frames_output/frames \
       data.val_dataset.manifest_path=/path/to/ego4d/manifests/val.json
   ```

### COCO

COCO already ships with hand-written captions, so no preprocessing
pipeline is needed — point the trainer directly at the
[Karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip)
JSON and the COCO 2014 train+val images:

```bash
torchrun --standalone --nproc-per-node=4 \
    -m apps.baselines.clip.training.train \
    name=coco_baseline \
    data=coco \
    data.train_dataset.manifest_path=/path/to/dataset_coco.json \
    data.train_dataset.image_root=/path/to/coco/all_images \
    data.val_dataset.manifest_path=/path/to/dataset_coco.json
```

The COCO loader (`apps.baselines.clip.data.CocoCaptionsDataset`)
understands the standard Karpathy schema (`{"images": [{"filename",
"sentences": [{"raw"|"tokens"}]}]}`). If you want a held-out validation
manifest, pre-split the Karpathy JSON into train-only / val-only files
and point each `*_dataset.manifest_path` at the matching split.

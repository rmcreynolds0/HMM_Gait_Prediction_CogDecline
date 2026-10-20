# Action Detection (APSC103 892C)

Real-time human action recognition and gait-deviation monitoring from video or a camera feed.
Built on **YOLO26n-pose** (pose estimation), **MMAction2 STGCN++** (skeleton action recognition), and a
custom **Gaussian HMM** gait analyser that learns a per-subject baseline and scores later walks against it.

---

## Architecture

```
                 ┌──────────────┐
                 │ Video/Camera │
                 └──────┬───────┘
                        │ BGR frame
                        ▼
              ┌──────────────────┐
              │  PoseEstimator   │  YOLO26n-pose → PoseResult (N persons × 17 COCO keypoints)
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ SkeletonConverter│  sliding window (window_size frames, fires every `stride`)
              └────────┬─────────┘
          window fires │ (same buffer snapshot feeds both branches)
            ┌──────────┴──────────────┐
            ▼                         ▼
   ┌──────────────────┐     ┌──────────────────────┐
   │ ActionRecognizer │     │ GaitFeatureExtractor │  → 15-dim GaitFeatureVector
   │ STGCN++ (NTU60)  │     └──────────┬───────────┘
   └────────┬─────────┘                ▼
            │               ┌──────────────────────┐
            │               │ GaitSessionManager   │  baseline collection → HMM training → scoring
            │               │  └─ GaitHMMAnalyser  │
            │               └──────────┬───────────┘
            │ Top-K actions            │ DeviationReport
            └────────────┬─────────────┘
                         ▼
                   FrameResult  → console callback + on-screen overlay
```

Both branches are driven by the same window trigger, so an action prediction and a gait report always
describe the same span of frames.

### Modules (`src/`)

| File | Responsibility |
|---|---|
| `main.py` | CLI entry point. Parses arguments, builds the pipeline, runs video or camera mode. |
| `visualize.py` | Alternate entry point with a richer overlay: per-person skeletons and IDs, action panel, gait panel, status bar. |
| `pipeline.py` | `ActionDetectionPipeline`: wires all components together, owns the frame loop, draws the default overlay, returns `FrameResult`s. |
| `pose_estimator.py` | `PoseEstimator` wraps ultralytics YOLO. Returns `PoseResult` with `keypoints (N,17,2)`, `keypoint_scores (N,17)`, `boxes (N,4)`, `box_scores (N,)`, `img_shape`. |
| `skeleton_converter.py` | `SkeletonConverter`: `deque`-based sliding window. Keeps the top `max_persons` by box score, zero-fills empty frames, emits MMAction2 `pose_results`. `get_raw_frames()` shares the buffer with the gait branch. |
| `action_recognizer.py` | `ActionRecognizer` wraps MMAction2 `inference_skeleton`. `recognize_top_k()` returns `[(label, score), ...]` over the 60 NTU60 classes (`NTU60_LABELS`). |
| `gait_feature_extractor.py` | `GaitFeatureExtractor`: turns a window of keypoints into one `GaitFeatureVector` (see features below). Picks the most confident person. Windows where ankles are visible in less than 40% of frames get a low-quality fallback vector. |
| `gait_session_manager.py` | `GaitSessionManager`: lifecycle state machine (`IDLE` → `COLLECTING_BASELINE` → `SCORING`), optional persistence of the trained model per subject. |
| `hmm_gait.py` | `GaussianHMM` (full covariance, log-space forward/backward, Baum-Welch EM, Viterbi) and `GaitHMMAnalyser` (baseline fitting, deviation scoring with bootstrapped 95% CI, alert levels). |
| `download_models.py` | Downloads STGCN++ weights to `models/` and YOLO26n-pose weights via ultralytics. |

### Gait analysis

**Features** (`hmm_gait.FEATURE_NAMES`, 15 per window): stride length, stride time, cadence, swing ratio,
stance ratio, double-support time, step width, trunk sway (AP and ML), heel-strike angle, toe-off angle,
centre-of-mass velocity, symmetry index, stride variability, smoothness (jerk). Distances are in pixels.

**Lifecycle:**
1. **Collecting baseline:** vectors are grouped into sessions of `--gait-cycles` vectors. After
   `--gait-sessions` sessions, a 4-state Gaussian HMM is trained on the subject's normal gait.
2. **Scoring:** every `--gait-score-every` new vectors, the running session is scored against the baseline
   log-likelihood. The resulting `DeviationReport` contains the deviation (σ), 95% CI, Viterbi state
   sequence, per-feature z-scores, confidence, and notes.
3. **Alert levels:** `normal` (CI lower bound < 0), `watch` (< 1.5σ), `alert` (< 3σ), `critical` (≥ 3σ).

**Persistence:** when `--baseline-dir` is set, the trained analyser is pickled to
`<baseline-dir>/<subject-id>_model.pkl` and reloaded on the next run. `--reset-baseline` wipes it.

### STGCN++ config (`configs/stgcnpp_config.py`)

`RecognizerGCN` with an `STGCN` backbone (COCO graph, spatial strategy, MSTCN temporal blocks) and a
`GCNHead` (60 classes). Test pipeline: `PreNormalize2D → GenSkeFeat → UniformSampleFrames(100) →
PoseDecode → FormatGCNInput(2 persons) → PackActionInputs`. Weights: `models/stgcnpp_ntu60_xsub_2d.pth`
(NTU RGB+D 60, XSub, 2D joint, OpenMMLab model zoo).

---

## Repository layout

```
action-detection/
├── README.md                 # this document
├── pyproject.toml / uv.lock  # dependencies (uv)
├── patch_mmaction.py         # one-time fix for mmaction2 1.2.0 missing `drn` module
├── verify_pipeline.py        # smoke test of every stage on synthetic data
├── video.mp4                 # sample input, also used by tests
├── yolo26n-pose.pt           # YOLO26n-pose weights
├── configs/stgcnpp_config.py
├── models/stgcnpp_ntu60_xsub_2d.pth
├── src/                      # core package (see Modules)
└── tests/                    # pytest suite
```

---

## Setup

Requirements: Python 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # 1. install dependencies
uv run python patch_mmaction.py          # 2. patch mmaction2 (idempotent, run once)
uv run python -m src.download_models     # 3. fetch weights (skipped if present)
uv run python verify_pipeline.py         # 4. smoke test, no video needed
```

## Running

```bash
# Video file
uv run python -m src.main --video video.mp4

# Camera
uv run python -m src.main --camera 0

# Gait baseline persisted for a subject
uv run python -m src.main --video walk.mp4 --subject-id P001 --baseline-dir ./baselines

# Action recognition only
uv run python -m src.main --camera 0 --no-gait

# Rich visualiser (same arguments, plus --kpt-threshold)
uv run python -m src.visualize --video walk.mp4 --subject-id P001 --baseline-dir ./baselines
```

Press `q` to quit the display window.

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--video` / `--camera` | required, one of | Video path or camera device ID |
| `--pose-model` | `yolo26n-pose.pt` | YOLO pose weights |
| `--action-config` | `configs/stgcnpp_config.py` | MMAction2 config |
| `--action-checkpoint` | `models/stgcnpp_ntu60_xsub_2d.pth` | STGCN++ weights |
| `--device` | `cpu` | `cpu` or `cuda:0` |
| `--window-size` | `48` | Sliding window length (frames) |
| `--stride` | `16` | Frames between inferences |
| `--pose-conf` | `0.5` | Pose detection confidence threshold |
| `--max-persons` | `2` | Persons kept per frame |
| `--top-k` | `5` | Action predictions returned |
| `--no-show` | off | Hide display window (video mode) |
| `--cam-width` / `--cam-height` | `720` / `1280` | Camera capture resolution |
| `--gait` / `--no-gait` | gait on | Enable/disable gait branch |
| `--subject-id` | `subject_000` | Subject key for baseline persistence |
| `--baseline-dir` | none | Where to save/load the trained baseline |
| `--gait-cycles` | `50` | Feature vectors per baseline session |
| `--gait-sessions` | `5` | Baseline sessions before training |
| `--gait-score-every` | `10` | Vectors between deviation reports |
| `--reset-baseline` | off | Discard persisted baseline and recollect |

## Testing

```bash
uv run pytest
```

Tests that need `yolo26n-pose.pt`, `models/stgcnpp_ntu60_xsub_2d.pth`, or `video.mp4` are skipped when
those files are missing.

## GPU

Point the PyTorch index in `pyproject.toml` at a CUDA build, re-sync, and pass `--device cuda:0`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cu118"   # or cu121
explicit = true
```

## Dependencies

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch / torchvision | 2.1.0 / 0.16.0 (CPU by default) |
| ultralytics | ≥ 8.4.12 |
| mmaction2 | 1.2.0 |
| mmcv-lite | ≥ 2.0, < 2.2 |
| mmengine | ≥ 0.10.7 |
| numpy | < 2.0 |
| scipy | HMM emission and statistics |

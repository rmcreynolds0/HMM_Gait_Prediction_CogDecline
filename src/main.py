"""CLI entry-point — Action + Gait Detection
============================================
Extends the original main.py with three new argument groups:

  Gait HMM arguments
  ------------------
  --gait              Enable the gait HMM deviation analyser (default: on).
  --no-gait           Disable the gait HMM branch entirely.
  --subject-id        Subject identifier used for model persistence.
  --baseline-dir      Directory to save/load the trained HMM baseline.
                      If omitted, baseline is not persisted across restarts.
  --gait-cycles       Feature vectors per baseline session   (default 50).
  --gait-sessions     Baseline sessions needed before training (default 5).
  --gait-score-every  Emit a DeviationReport every N vectors  (default 10).
  --reset-baseline    Wipe any persisted baseline and start fresh.

All original arguments are unchanged for drop-in compatibility.

Usage examples
--------------
  # Process a video with gait analysis, saving baseline for subject P001
  uv run python -m src.main --video walk.mp4 --subject-id P001 \\
      --baseline-dir ./baselines

  # Camera mode, gait disabled
  uv run python -m src.main --camera 0 --no-gait

  # Reset stored baseline and recollect
  uv run python -m src.main --video walk.mp4 --subject-id P001 \\
      --baseline-dir ./baselines --reset-baseline
"""

import argparse
import sys
from pathlib import Path


def print_prediction(frame_idx: int, predictions: list) -> None:
    """Terminal callback for action recognition results."""
    print(f"\n[Frame {frame_idx}] Action recognition:")
    for i, (label, score) in enumerate(predictions, 1):
        print(f"  {i}. {label}: {score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Action + Gait Detection: YOLO26n-pose + STGCN++ + Gait HMM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input source ─────────────────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video",  type=str, help="Video file path")
    input_group.add_argument("--camera", type=int, help="Camera device ID")

    # ── Model paths ───────────────────────────────────────────────────────────
    parser.add_argument("--pose-model",         type=str,
                        default="yolo26n-pose.pt",
                        help="YOLO pose model path")
    parser.add_argument("--action-config",      type=str,
                        default="configs/stgcnpp_config.py",
                        help="MMAction2 config file path")
    parser.add_argument("--action-checkpoint",  type=str,
                        default="models/stgcnpp_ntu60_xsub_2d.pth",
                        help="MMAction2 model weights path")

    # ── Inference parameters ──────────────────────────────────────────────────
    parser.add_argument("--device",       type=str,   default="cpu",
                        help="Inference device (cpu or cuda:0)")
    parser.add_argument("--window-size",  type=int,   default=48,
                        help="Sliding window size (frames)")
    parser.add_argument("--stride",       type=int,   default=16,
                        help="Window sliding stride")
    parser.add_argument("--pose-conf",    type=float, default=0.5,
                        help="Pose detection confidence threshold")
    parser.add_argument("--max-persons",  type=int,   default=2,
                        help="Max tracked persons")
    parser.add_argument("--top-k",        type=int,   default=5,
                        help="Top-K action recognition results")
    parser.add_argument("--no-show",      action="store_true",
                        help="Disable visualisation window (video mode)")
    parser.add_argument("--cam-width",    type=int,   default=720,
                        help="Camera capture width")
    parser.add_argument("--cam-height",   type=int,   default=1280,
                        help="Camera capture height")

    # ── NEW: Gait HMM arguments ───────────────────────────────────────────────
    gait_group = parser.add_argument_group("Gait HMM")
    gait_excl  = gait_group.add_mutually_exclusive_group()
    gait_excl.add_argument("--gait",    dest="enable_gait",
                           action="store_true",  default=True,
                           help="Enable gait HMM branch (default)")
    gait_excl.add_argument("--no-gait", dest="enable_gait",
                           action="store_false",
                           help="Disable gait HMM branch")
    gait_group.add_argument("--subject-id",       type=str,
                             default="subject_000",
                             help="Subject identifier (used for model persistence)")
    gait_group.add_argument("--baseline-dir",     type=str,
                             default=None,
                             help="Directory for persisting trained HMM baseline")
    gait_group.add_argument("--gait-cycles",      type=int, default=50,
                             help="Feature vectors per baseline session")
    gait_group.add_argument("--gait-sessions",    type=int, default=5,
                             help="Baseline sessions required before HMM training")
    gait_group.add_argument("--gait-score-every", type=int, default=10,
                             help="Emit a DeviationReport every N feature vectors")
    gait_group.add_argument("--reset-baseline",   action="store_true",
                             default=False,
                             help="Delete persisted baseline and recollect from scratch")

    args = parser.parse_args()

    # ── Guard: STGCN++ checkpoint ─────────────────────────────────────────────
    checkpoint = Path(args.action_checkpoint)
    if not checkpoint.exists():
        print(f"[Error] STGCN++ weights not found: {checkpoint}")
        print("  Run: uv run python -m src.download_models")
        sys.exit(1)

    from .pipeline import ActionDetectionPipeline

    def on_gait_report(report) -> None:
        print(f"\n[Gait] Session {report.session_id}  "
              f"deviation={report.deviation_score:+.3f}σ  "
              f"CI=[{report.ci_lower:.2f}, {report.ci_upper:.2f}]  "
              f"alert={report.alert_level.upper()}")
        for note in report.notes:
            print(f"  • {note}")

    pipeline = ActionDetectionPipeline(
        # original args
        pose_model_path=args.pose_model,
        action_config_path=args.action_config,
        action_checkpoint_path=args.action_checkpoint,
        device=args.device,
        window_size=args.window_size,
        stride=args.stride,
        pose_conf_threshold=args.pose_conf,
        max_persons=args.max_persons,
        top_k=args.top_k,
        # gait args
        enable_gait=args.enable_gait,
        subject_id=args.subject_id,
        baseline_dir=args.baseline_dir,
        gait_cycles_per_session=args.gait_cycles,
        gait_min_sessions=args.gait_sessions,
        gait_score_every=args.gait_score_every,
        on_gait_report=on_gait_report if args.enable_gait else None,
    )

    # Optionally wipe baseline before starting
    if args.reset_baseline and args.enable_gait:
        pipeline.reset_gait_baseline()

    # ── Run ───────────────────────────────────────────────────────────────────
    if args.video:
        vpath = Path(args.video)
        if not vpath.exists():
            print(f"[Error] Video not found: {vpath}")
            sys.exit(1)
        print(f"[Main] Processing video: {vpath}")
        results = pipeline.process_video(
            str(vpath),
            callback=print_prediction,
            show=not args.no_show,
        )
        print(f"\n[Main] Done. {len(results)} window inferences.")

    elif args.camera is not None:
        print(f"[Main] Opening camera: {args.camera}")
        pipeline.process_camera(
            camera_id=args.camera,
            callback=print_prediction,
            cam_width=args.cam_width,
            cam_height=args.cam_height,
        )


if __name__ == "__main__":
    main()

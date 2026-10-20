"""Enhanced Visualiser — Skeleton + Action + Gait HMM Overlay
==============================================================
Extends the original visualize.py with:

  1. Gait deviation panel  — top-right corner showing:
       • Current mode (Collecting Baseline / Scoring)
       • Deviation score in σ with 95 % CI
       • Alert level badge (colour-coded)
       • Animated progress bar (baseline collection OR deviation magnitude)
       • Top-contributing feature and its z-score

  2. Per-person skeleton rendering and bounding-box tracking IDs
     (unchanged from original)

  3. Action recognition panel (unchanged from original)

  4. Status bar at the bottom (unchanged from original)

All new args mirror those in main.py (--gait / --no-gait / --subject-id /
--baseline-dir / --gait-cycles / --gait-sessions / --gait-score-every /
--reset-baseline).

Usage
-----
  uv run python -m src.visualize --video walk.mp4 \\
      --subject-id P001 --baseline-dir ./baselines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .action_recognizer import ActionRecognizer
from .pose_estimator import PoseEstimator, PoseResult
from .skeleton_converter import SkeletonConverter
from .gait_feature_extractor import GaitFeatureExtractor
from .gait_session_manager import GaitSessionManager, GaitMode
from .hmm_gait import DeviationReport

# ── COCO skeleton ─────────────────────────────────────────────────────────────
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]

PERSON_COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255),
    (255, 0, 255), (0, 255, 255), (255, 255, 0),
]

# Alert → BGR colour
ALERT_BGR: Dict[str, Tuple[int, int, int]] = {
    "normal":   (50,  200,  50),
    "watch":    (30,  200, 200),
    "alert":    (30,  100, 240),
    "critical": (40,   40, 220),
}


def _person_color(pid: int) -> Tuple[int, int, int]:
    return PERSON_COLORS[pid % len(PERSON_COLORS)]


# ── Skeleton / bbox rendering ─────────────────────────────────────────────────

def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    color: Tuple[int, int, int],
    kpt_threshold: float = 0.3,
) -> None:
    for i, j in SKELETON_EDGES:
        if scores[i] > kpt_threshold and scores[j] > kpt_threshold:
            pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
    for k in range(17):
        if scores[k] > kpt_threshold:
            cx, cy = int(keypoints[k, 0]), int(keypoints[k, 1])
            cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_bbox_with_id(
    frame: np.ndarray,
    box: np.ndarray,
    person_id: int,
    confidence: float,
    color: Tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{person_id} ({confidence:.2f})"
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# ── Action recognition panel ──────────────────────────────────────────────────

def draw_action_panel(
    frame: np.ndarray,
    predictions: List[Tuple[str, float]],
) -> None:
    if not predictions:
        return
    panel_h = 32 + 24 * len(predictions)
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (385, 5 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "Action Recognition", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1, cv2.LINE_AA)
    for i, (label, score) in enumerate(predictions):
        y = 46 + i * 24
        bar_len = int(score * 180)
        cv2.rectangle(frame, (10, y - 10), (10 + bar_len, y + 4), (0, 180, 0), -1)
        cv2.putText(frame, f"{label}: {score:.3f}", (10, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


# ── NEW: Gait HMM panel ───────────────────────────────────────────────────────

def draw_gait_panel(
    frame: np.ndarray,
    gait_manager: GaitSessionManager,
    latest_report: Optional[DeviationReport],
) -> None:
    """
    Render a gait deviation panel in the top-right corner.

    Layout:
      ┌──────────────────────────────┐
      │ GAIT HMM  ● ALERT_LEVEL      │  ← header + badge
      │ ▓▓▓▓▓▓░░░░░░  +2.34σ        │  ← deviation bar + score
      │ 95% CI: [1.20, 3.48]         │  ← confidence interval
      │ Top: double_support +3.1SD   │  ← top contributing feature
      │ Conf: 87%  Cycles: 42        │  ← metadata
      └──────────────────────────────┘
    or during collection:
      ┌──────────────────────────────┐
      │ GAIT HMM  COLLECTING...      │
      │ ▓▓▓░░░░░░░░░░  38%           │  ← baseline progress bar
      │ Walk normally to collect     │
      │ 3 / 5 sessions complete      │
      └──────────────────────────────┘
    """
    h, w = frame.shape[:2]
    panel_w, panel_h = 310, 110
    x0 = w - panel_w - 8
    y0 = 8

    report = latest_report or gait_manager.last_report
    mode   = gait_manager.mode

    # Choose accent colour
    if mode == GaitMode.COLLECTING_BASELINE:
        accent = (100, 200, 100)
    elif report:
        accent = ALERT_BGR.get(report.alert_level, (180, 180, 180))
    else:
        accent = (150, 150, 150)

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), accent, 2)

    line_h = 18
    tx, ty = x0 + 7, y0 + 16

    # ── Header ────────────────────────────────────────────────────────────────
    cv2.putText(frame, "GAIT HMM", (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    if mode == GaitMode.COLLECTING_BASELINE:
        badge_text = "COLLECTING"
        badge_color = (100, 200, 100)
    elif report:
        badge_text = f"● {report.alert_level.upper()}"
        badge_color = accent
    else:
        badge_text = "READY"
        badge_color = (150, 150, 150)

    cv2.putText(frame, badge_text, (x0 + 130, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, badge_color, 1, cv2.LINE_AA)

    ty += line_h + 2
    bar_max_w = panel_w - 14

    # ── Main bar ─────────────────────────────────────────────────────────────
    if mode == GaitMode.COLLECTING_BASELINE:
        pct = gait_manager.baseline_progress
        bw  = int(bar_max_w * pct)
        cv2.rectangle(frame, (tx, ty), (tx + bw, ty + 14), (80, 160, 80), -1)
        cv2.rectangle(frame, (tx, ty), (tx + bar_max_w, ty + 14), (60, 60, 60), 1)
        cv2.putText(frame, f"{int(pct*100)}%", (tx + bar_max_w + 4, ty + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 180), 1)
    elif report:
        clamped = min(abs(report.deviation_score) / 6.0, 1.0)
        bw = int(bar_max_w * clamped)
        cv2.rectangle(frame, (tx, ty), (tx + bw, ty + 14), accent, -1)
        cv2.rectangle(frame, (tx, ty), (tx + bar_max_w, ty + 14), (60, 60, 60), 1)
        score_txt = f"{report.deviation_score:+.2f}σ"
        cv2.putText(frame, score_txt, (tx + bar_max_w + 4, ty + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, accent, 1)

    ty += line_h + 4

    # ── Details ───────────────────────────────────────────────────────────────
    if mode == GaitMode.COLLECTING_BASELINE:
        n_sessions = len(gait_manager._baseline_sessions)
        total      = gait_manager.min_baseline_sessions
        cv2.putText(frame, "Walk normally to build baseline",
                    (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (160, 210, 160), 1, cv2.LINE_AA)
        ty += line_h
        cv2.putText(frame, f"Sessions: {n_sessions}/{total}",
                    (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (160, 210, 160), 1, cv2.LINE_AA)
        ty += line_h
        buf_needed = (gait_manager.cycles_per_baseline_session
                      - len(gait_manager._current_baseline_buf))
        cv2.putText(frame, f"Next session: {buf_needed} cycles remaining",
                    (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (120, 170, 120), 1, cv2.LINE_AA)

    elif report:
        cv2.putText(frame,
                    f"95% CI: [{report.ci_lower:.2f}, {report.ci_upper:.2f}]",
                    (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (180, 180, 180), 1, cv2.LINE_AA)
        ty += line_h

        top_feat, top_z = max(
            report.per_feature_deviation.items(),
            key=lambda kv: abs(kv[1]),
            default=("–", 0.0),
        )
        feat_label = top_feat.replace("_", " ")
        cv2.putText(frame,
                    f"Top: {feat_label} {top_z:+.2f}SD",
                    (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (200, 180, 140), 1, cv2.LINE_AA)
        ty += line_h

        cv2.putText(frame,
                    f"Conf: {report.confidence:.0%}   Cycles: {report.n_cycles}",
                    (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (140, 140, 140), 1, cv2.LINE_AA)


# ── Status bar ────────────────────────────────────────────────────────────────

def draw_status_bar(
    frame: np.ndarray,
    frame_idx: int,
    n_persons: int,
    buffer_size: int,
    window_size: int,
    gait_mode: str = "",
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    status = (f"Frame: {frame_idx}  |  Persons: {n_persons}  |  "
              f"Buffer: {buffer_size}/{window_size}  |  "
              f"Gait: {gait_mode}  |  Press 'q' to quit")
    cv2.putText(frame, status, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


# ── Terminal callback ─────────────────────────────────────────────────────────

def print_prediction(frame_idx: int, predictions: list) -> None:
    print(f"\n[Frame {frame_idx}] Action recognition:")
    for i, (label, score) in enumerate(predictions, 1):
        print(f"  {i}. {label}: {score:.4f}")


# ── Main visualisation loop ───────────────────────────────────────────────────

def run_visualize(args: argparse.Namespace) -> None:
    print("[Visualize] Loading YOLO26n-pose model...")
    pose_estimator = PoseEstimator(
        model_path=args.pose_model,
        device=args.device,
        conf_threshold=args.pose_conf,
    )

    print("[Visualize] Loading STGCN++ model...")
    action_recognizer = ActionRecognizer(
        config_path=args.action_config,
        checkpoint_path=args.action_checkpoint,
        device=args.device,
    )

    converter = SkeletonConverter(
        window_size=args.window_size,
        stride=args.stride,
        max_persons=args.max_persons,
    )

    # ── Gait subsystems ───────────────────────────────────────────────────────
    gait_extractor: Optional[GaitFeatureExtractor] = None
    gait_manager:   Optional[GaitSessionManager]   = None
    latest_gait_report: Optional[DeviationReport]  = None

    if args.enable_gait:
        fps_hint = 30.0  # updated from VideoCapture below
        gait_extractor = GaitFeatureExtractor(fps=fps_hint)

        def on_report(report: DeviationReport) -> None:
            nonlocal latest_gait_report
            latest_gait_report = report
            print(f"\n[Gait] {report.session_id}  "
                  f"{report.deviation_score:+.3f}σ  "
                  f"CI=[{report.ci_lower:.2f},{report.ci_upper:.2f}]  "
                  f"alert={report.alert_level.upper()}")
            for note in report.notes:
                print(f"  • {note}")

        gait_manager = GaitSessionManager(
            subject_id=args.subject_id,
            baseline_dir=args.baseline_dir,
            cycles_per_baseline_session=args.gait_cycles,
            min_baseline_sessions=args.gait_sessions,
            score_every_n=args.gait_score_every,
            on_report=on_report,
        )
        if args.reset_baseline:
            gait_manager.reset_baseline()

    print("[Visualize] Initialisation complete.")

    # ── Open source ───────────────────────────────────────────────────────────
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if args.enable_gait:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            gait_extractor.fps = fps  # type: ignore[union-attr]
    else:
        cap = cv2.VideoCapture(args.camera)
        if args.cam_width and args.cam_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Visualize] Camera resolution: {aw}×{ah}")

    if not cap.isOpened():
        source = args.video or args.camera
        print(f"[Error] Cannot open source: {source}")
        sys.exit(1)

    frame_idx = 0
    latest_predictions: List[Tuple[str, float]] = []
    total_recognitions = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    break
                print("[Visualize] Camera read failed.")
                break

            # ── Pose estimation ───────────────────────────────────────────────
            pose_result = pose_estimator.predict_track(frame)
            should_infer = converter.add_frame(pose_result)

            if should_infer:
                # ── Action recognition ────────────────────────────────────────
                data = converter.get_pose_results()
                if data is not None:
                    pose_results_list, img_shape = data
                    latest_predictions = action_recognizer.recognize_top_k(
                        pose_results_list, img_shape, top_k=args.top_k)
                    total_recognitions += 1
                    print_prediction(frame_idx, latest_predictions)

                # ── Gait HMM ──────────────────────────────────────────────────
                if args.enable_gait and gait_extractor and gait_manager:
                    raw_frames = converter.get_raw_frames()
                    if raw_frames:
                        ts  = frame_idx / max(gait_extractor.fps, 1.0)
                        fv  = gait_extractor.extract(raw_frames, timestamp=ts)
                        if fv is not None:
                            report = gait_manager.add_vector(fv)
                            if report is not None:
                                latest_gait_report = report

            # ── Rendering ─────────────────────────────────────────────────────
            display    = frame.copy()
            n_persons  = pose_result.keypoints.shape[0]

            # Skeletons + bounding boxes
            for p in range(n_persons):
                pid = (int(pose_result.track_ids[p])
                       if pose_result.track_ids is not None else p)
                color = _person_color(pid)
                draw_skeleton(display,
                              pose_result.keypoints[p],
                              pose_result.keypoint_scores[p],
                              color, kpt_threshold=args.kpt_threshold)
                draw_bbox_with_id(display, pose_result.boxes[p],
                                  pid, float(pose_result.box_scores[p]), color)

            # Action panel (top-left)
            draw_action_panel(display, latest_predictions)

            # Gait panel (top-right)
            if args.enable_gait and gait_manager:
                draw_gait_panel(display, gait_manager, latest_gait_report)

            # Status bar (bottom)
            gait_mode_str = (gait_manager.mode.name
                             if gait_manager else "disabled")
            draw_status_bar(display, frame_idx, n_persons,
                            converter.buffer_size, converter.window_size,
                            gait_mode_str)

            # Scale down for display if frame is too tall
            max_h = 900
            dh, dw = display.shape[:2]
            if dh > max_h:
                scale = max_h / dh
                display = cv2.resize(display, (int(dw * scale), max_h))

            cv2.imshow("Action + Gait Detection — Skeleton Visualiser", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n[Visualize] Done. {frame_idx} frames, "
          f"{total_recognitions} action inferences.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Action + Gait Visualiser — Skeleton + Tracking IDs + HMM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video",  type=str, help="Video file path")
    input_group.add_argument("--camera", type=int, help="Camera device ID")

    parser.add_argument("--pose-model",        type=str,
                        default="yolo26n-pose.pt")
    parser.add_argument("--action-config",     type=str,
                        default="configs/stgcnpp_config.py")
    parser.add_argument("--action-checkpoint", type=str,
                        default="models/stgcnpp_ntu60_xsub_2d.pth")
    parser.add_argument("--device",            type=str,  default="cpu")
    parser.add_argument("--window-size",       type=int,  default=48)
    parser.add_argument("--stride",            type=int,  default=16)
    parser.add_argument("--pose-conf",         type=float, default=0.5)
    parser.add_argument("--kpt-threshold",     type=float, default=0.3)
    parser.add_argument("--max-persons",       type=int,  default=2)
    parser.add_argument("--top-k",             type=int,  default=5)
    parser.add_argument("--cam-width",         type=int,  default=720)
    parser.add_argument("--cam-height",        type=int,  default=1280)

    # Gait args
    gait_grp  = parser.add_argument_group("Gait HMM")
    gait_excl = gait_grp.add_mutually_exclusive_group()
    gait_excl.add_argument("--gait",    dest="enable_gait",
                           action="store_true", default=True)
    gait_excl.add_argument("--no-gait", dest="enable_gait",
                           action="store_false")
    gait_grp.add_argument("--subject-id",       type=str,  default="subject_000")
    gait_grp.add_argument("--baseline-dir",     type=str,  default=None)
    gait_grp.add_argument("--gait-cycles",      type=int,  default=50)
    gait_grp.add_argument("--gait-sessions",    type=int,  default=5)
    gait_grp.add_argument("--gait-score-every", type=int,  default=10)
    gait_grp.add_argument("--reset-baseline",   action="store_true", default=False)

    args = parser.parse_args()

    if not Path(args.action_checkpoint).exists():
        print(f"[Error] Checkpoint not found: {args.action_checkpoint}")
        print("  Run: uv run python -m src.download_models")
        sys.exit(1)

    if args.video and not Path(args.video).exists():
        print(f"[Error] Video not found: {args.video}")
        sys.exit(1)

    run_visualize(args)


if __name__ == "__main__":
    main()

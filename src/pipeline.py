"""Integrated Action + Gait Detection Pipeline
===============================================
Extends the original YOLO26n-pose → SkeletonConverter → STGCN++ action
recognition pipeline with a parallel gait-HMM analysis branch.

New data-flow after this change:

  ┌──────────────┐
  │ Video/Camera │
  └──────┬───────┘
         │ raw frame (BGR)
         ▼
  ┌──────────────────────┐
  │  PoseEstimator        │  YOLO26n-pose → PoseResult (N, 17, 2)
  └──────┬───────────────┘
         │ PoseResult
         ├────────────────────────────────────────────────────────┐
         ▼                                                        ▼
  ┌─────────────────────┐                         ┌──────────────────────────┐
  │  SkeletonConverter   │  (unchanged)            │  GaitFeatureExtractor    │
  │  (sliding window)    │                         │  (same window, parallel) │
  └──────┬──────────────┘                         └──────────┬───────────────┘
         │ pose_results (MMAction2 fmt)                      │ GaitFeatureVector
         ▼                                                   ▼
  ┌──────────────┐                                ┌──────────────────────────┐
  │  STGCN++     │                                │  GaitSessionManager      │
  │  ActionRec.  │                                │  (HMM baseline/scoring)  │
  └──────┬───────┘                                └──────────┬───────────────┘
         │ Top-K actions                                     │ DeviationReport
         └─────────────────────────┬─────────────────────────┘
                                   ▼
                         FrameResult (combined output)

Notes on synchronisation
------------------------
Both branches share the same window trigger: when SkeletonConverter.add_frame()
returns True, we simultaneously call GaitFeatureExtractor.extract() on the same
buffer snapshot.  This guarantees both outputs refer to the same temporal window.

Changes to existing modules
----------------------------
- skeleton_converter.py: expose ``get_raw_frames()`` to share the buffer with
  the gait extractor without duplicating state.
- pipeline.py: this file — complete rewrite to add gait branch.
- main.py: new ``--gait`` / ``--subject-id`` / ``--baseline-dir`` args.
- visualize.py: render gait overlay panel alongside action panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .action_recognizer import ActionRecognizer
from .pose_estimator import PoseEstimator
from .skeleton_converter import SkeletonConverter
from .gait_feature_extractor import GaitFeatureExtractor
from .gait_session_manager import GaitSessionManager, GaitMode
from .hmm_gait import DeviationReport


# ── Combined per-window result ────────────────────────────────────────────────

@dataclass
class FrameResult:
    """Combined output produced once per sliding-window trigger.

    Attributes:
        frame_idx:       Index of the last frame in the window.
        predictions:     Top-K action recognition results [(label, score), ...].
        gait_report:     HMM deviation report, or None if gait is disabled /
                         still collecting baseline.
        gait_status:     Human-readable gait status string for overlay.
        n_persons:       Number of persons detected in the last frame.
    """
    frame_idx:    int
    predictions:  List[Tuple[str, float]]
    gait_report:  Optional[DeviationReport] = None
    gait_status:  str = ""
    n_persons:    int = 0


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ActionDetectionPipeline:
    """
    Action + Gait detection pipeline.

    Args
    ----
    pose_model_path, action_config_path, action_checkpoint_path, device,
    window_size, stride, pose_conf_threshold, max_persons, top_k:
        Identical to the original pipeline; see original documentation.

    enable_gait: bool
        When True, the gait HMM branch is activated.  Defaults to True.
    subject_id: str
        Subject identifier passed to GaitSessionManager.
    baseline_dir: Optional[str]
        Directory for persisting the trained HMM.  None disables persistence.
    gait_fps: float
        Frames-per-second of the source, used for gait cadence / timing.
        Pipeline auto-detects from cv2.VideoCapture when processing a file;
        override here for camera streams.
    gait_cycles_per_session: int
        Feature vectors (windows) per baseline session.
    gait_min_sessions: int
        Baseline sessions required before HMM training begins.
    gait_score_every: int
        Emit a DeviationReport every N feature vectors during scoring.
    on_gait_report: Optional[Callable[[DeviationReport], None]]
        Callback invoked each time a new gait deviation report is produced.
    """

    def __init__(
        self,
        # ── original args ──
        pose_model_path: str = "yolo26n-pose.pt",
        action_config_path: str = "configs/stgcnpp_config.py",
        action_checkpoint_path: str = "models/stgcnpp_ntu60_xsub_2d.pth",
        device: str = "cpu",
        window_size: int = 48,
        stride: int = 16,
        pose_conf_threshold: float = 0.5,
        max_persons: int = 2,
        top_k: int = 5,
        # ── new gait args ──
        enable_gait: bool = True,
        subject_id: str = "subject_000",
        baseline_dir: Optional[str] = None,
        gait_fps: float = 30.0,
        gait_cycles_per_session: int = 50,
        gait_min_sessions: int = 5,
        gait_score_every: int = 10,
        on_gait_report: Optional[Callable[[DeviationReport], None]] = None,
    ) -> None:
        self.device      = device
        self.top_k       = top_k
        self.enable_gait = enable_gait
        self._gait_fps   = gait_fps

        # ── Action recognition sub-systems ───────────────────────────────────
        print("[Pipeline] Loading YOLO26n-pose model...")
        self.pose_estimator = PoseEstimator(
            model_path=pose_model_path,
            device=device,
            conf_threshold=pose_conf_threshold,
        )

        print("[Pipeline] Loading STGCN++ model...")
        self.action_recognizer = ActionRecognizer(
            config_path=action_config_path,
            checkpoint_path=action_checkpoint_path,
            device=device,
        )

        self.converter = SkeletonConverter(
            window_size=window_size,
            stride=stride,
            max_persons=max_persons,
        )

        # ── Gait HMM sub-systems ─────────────────────────────────────────────
        if self.enable_gait:
            print("[Pipeline] Initialising gait HMM analyser...")
            self.gait_extractor = GaitFeatureExtractor(fps=gait_fps)
            self.gait_manager   = GaitSessionManager(
                subject_id=subject_id,
                baseline_dir=baseline_dir,
                cycles_per_baseline_session=gait_cycles_per_session,
                min_baseline_sessions=gait_min_sessions,
                score_every_n=gait_score_every,
                on_report=on_gait_report,
                on_baseline_ready=self._on_baseline_ready,
            )
        else:
            self.gait_extractor = None  # type: ignore[assignment]
            self.gait_manager   = None  # type: ignore[assignment]

        print("[Pipeline] Initialisation complete.")

    # ── Video / camera processing ─────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        callback: Optional[Callable] = None,
        show: bool = False,
    ) -> List[FrameResult]:
        """Process a video file end-to-end.

        Args:
            video_path: Path to the video file.
            callback: Optional callable invoked per FrameResult:
                      ``callback(frame_idx, predictions)`` — kept signature-
                      compatible with original pipeline for drop-in use.
            show: Display annotated video in a window.

        Returns:
            List of FrameResult, one per window trigger.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Auto-detect FPS and update gait extractor
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.enable_gait and fps > 0:
            self.gait_extractor.fps = fps

        results: List[FrameResult] = []
        frame_idx = 0
        latest_result = FrameResult(frame_idx=0, predictions=[])

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_result = self._process_frame(frame, frame_idx)
                if frame_result is not None:
                    latest_result = frame_result
                    results.append(frame_result)
                    if callback:
                        # backward-compatible signature
                        callback(frame_idx, frame_result.predictions)

                if show:
                    display = self._draw_annotations(frame, latest_result)
                    cv2.imshow("Action + Gait Detection", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()

        return results

    def process_camera(
        self,
        camera_id: int = 0,
        callback: Optional[Callable] = None,
        cam_width: Optional[int] = None,
        cam_height: Optional[int] = None,
    ) -> None:
        """Process a live camera stream.

        Args identical to original pipeline; gait analysis runs automatically
        in the background.
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")

        if cam_width is not None and cam_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

        frame_idx = 0
        latest_result = FrameResult(frame_idx=0, predictions=[])

        try:
            print("[Pipeline] Press 'q' to exit camera mode...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Pipeline] Camera read failed.")
                    break

                frame_result = self._process_frame(frame, frame_idx)
                if frame_result is not None:
                    latest_result = frame_result
                    if callback:
                        callback(frame_idx, frame_result.predictions)

                display = self._draw_annotations(frame, latest_result)
                cv2.imshow("Action + Gait Detection — Camera", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_idx += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()

    # ── Core per-frame logic ──────────────────────────────────────────────────

    def _process_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> Optional[FrameResult]:
        """
        Process one frame through both the action-recognition and gait branches.

        Returns a FrameResult when the sliding window triggers, else None.
        """
        h, w = frame.shape[:2]

        # 1. Pose estimation (shared between both branches)
        pose_result = self.pose_estimator.predict(frame)

        # 2. Add to sliding-window converter
        should_infer = self.converter.add_frame(pose_result)

        if not should_infer:
            return None

        # ── Branch A: Action recognition (STGCN++) ───────────────────────────
        predictions: List[Tuple[str, float]] = []
        stgcn_data = self.converter.get_pose_results()
        if stgcn_data is not None:
            pose_results_list, img_shape = stgcn_data
            predictions = self.action_recognizer.recognize_top_k(
                pose_results_list, img_shape, top_k=self.top_k
            )

        # ── Branch B: Gait HMM ───────────────────────────────────────────────
        gait_report: Optional[DeviationReport] = None
        gait_status: str = ""

        if self.enable_gait:
            # Retrieve the same raw frames the converter holds
            raw_frames = self.converter.get_raw_frames()
            if raw_frames:
                timestamp = frame_idx / max(self._gait_fps, 1.0)
                fv = self.gait_extractor.extract(raw_frames, timestamp=timestamp)
                if fv is not None:
                    report = self.gait_manager.add_vector(fv)
                    if report is not None:
                        gait_report = report
            gait_status = self.gait_manager.get_status_string()

        return FrameResult(
            frame_idx=frame_idx,
            predictions=predictions,
            gait_report=gait_report,
            gait_status=gait_status,
            n_persons=pose_result.keypoints.shape[0],
        )

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _draw_annotations(
        self, frame: np.ndarray, result: FrameResult
    ) -> np.ndarray:
        """Render action predictions and gait overlay onto the frame."""
        display = frame.copy()

        # ── Action recognition panel (unchanged from original) ────────────────
        if result.predictions:
            overlay = display.copy()
            panel_h = 30 + 25 * len(result.predictions)
            cv2.rectangle(overlay, (5, 5), (400, panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        for i, (label, score) in enumerate(result.predictions):
            text = f"{label}: {score:.3f}"
            cv2.putText(display, text, (10, 28 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ── Gait HMM overlay ─────────────────────────────────────────────────
        if self.enable_gait:
            self._draw_gait_overlay(display, result)

        # ── Buffer progress bar ───────────────────────────────────────────────
        buf  = self.converter.buffer_size
        win  = self.converter.window_size
        h    = display.shape[0]
        cv2.putText(display, f"Buffer: {buf}/{win}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        return display

    def _draw_gait_overlay(
        self, frame: np.ndarray, result: FrameResult
    ) -> None:
        """
        Draw gait deviation panel in the top-right corner of the frame.
        Panel colour encodes alert level:
          normal → green  |  watch → yellow  |  alert → orange  |  critical → red
        """
        h, w = frame.shape[:2]
        panel_w, panel_h = 300, 95
        x0, y0 = w - panel_w - 8, 8

        # Alert → colour mapping (BGR)
        alert_colors: Dict[str, Tuple[int, int, int]] = {
            "normal":   (50, 200, 50),
            "watch":    (30, 200, 200),
            "alert":    (30, 100, 240),
            "critical": (40, 40, 220),
        }

        report = result.gait_report or self.gait_manager.last_report
        alert_level = report.alert_level if report else "normal"
        color = alert_colors.get(alert_level, (180, 180, 180))

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Coloured border
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      color, 2)

        # Status line
        cv2.putText(frame, result.gait_status or "GAIT: --",
                    (x0 + 6, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        if report:
            # Deviation bar
            bar_max_w = panel_w - 16
            clamped   = min(abs(report.deviation_score) / 5.0, 1.0)
            bar_w     = int(bar_max_w * clamped)
            cv2.rectangle(frame,
                          (x0 + 8, y0 + 32),
                          (x0 + 8 + bar_w, y0 + 48),
                          color, -1)
            cv2.rectangle(frame,
                          (x0 + 8, y0 + 32),
                          (x0 + 8 + bar_max_w, y0 + 48),
                          (80, 80, 80), 1)

            # Per-window details
            top_feat = max(report.per_feature_deviation.items(),
                           key=lambda kv: abs(kv[1]),
                           default=("–", 0.0))
            cv2.putText(frame,
                        f"Top feat: {top_feat[0]} ({top_feat[1]:+.2f}SD)",
                        (x0 + 6, y0 + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame,
                        f"Conf: {report.confidence:.0%}  "
                        f"Cycles: {report.n_cycles}",
                        (x0 + 6, y0 + 83),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (160, 160, 160), 1, cv2.LINE_AA)
        else:
            # Collecting baseline
            pct = int(self.gait_manager.baseline_progress * 100)
            bar_w = int((panel_w - 16) * pct / 100)
            cv2.rectangle(frame,
                          (x0 + 8, y0 + 35),
                          (x0 + 8 + bar_w, y0 + 50),
                          (100, 180, 100), -1)
            cv2.rectangle(frame,
                          (x0 + 8, y0 + 35),
                          (x0 + panel_w - 8, y0 + 50),
                          (80, 80, 80), 1)
            cv2.putText(frame,
                        f"Collecting baseline... {pct}%",
                        (x0 + 6, y0 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (160, 200, 160), 1, cv2.LINE_AA)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset both action-recognition and gait pipeline state."""
        self.converter.reset()
        if self.enable_gait:
            self.gait_extractor.reset()
            self.gait_manager.reset_session()

    def reset_gait_baseline(self) -> None:
        """Wipe gait baseline and restart collection from scratch."""
        if self.enable_gait:
            self.gait_manager.reset_baseline()

    # ── Internal callbacks ────────────────────────────────────────────────────

    @staticmethod
    def _on_baseline_ready() -> None:
        print("[Pipeline] ✓ Gait baseline training complete — now scoring.")

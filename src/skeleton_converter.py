"""Skeleton Converter — Sliding Window Buffer
=============================================
Unchanged from the original except for one addition:

  ``get_raw_frames() -> list[PoseResult]``
      Returns a snapshot of the current window as a plain list of PoseResult
      objects.  Used by GaitFeatureExtractor to extract gait features from the
      same temporal window without duplicating buffer state.

All original behaviour (add_frame, get_pose_results, reset, is_ready,
buffer_size) is preserved exactly.
"""

from collections import deque
from typing import List, Optional

import numpy as np

from .pose_estimator import PoseResult


class SkeletonConverter:
    """Sliding-window skeleton data converter.

    Accumulates per-frame YOLO pose results and converts them into the
    MMAction2 ``inference_skeleton`` format when the window is full.

    Args:
        window_size: Sliding window size in frames (default 48).
        stride: Window sliding stride (default 16).
        max_persons: Maximum persons to retain per frame (default 2).
    """

    def __init__(
        self,
        window_size: int = 48,
        stride: int = 16,
        max_persons: int = 2,
    ) -> None:
        self.window_size = window_size
        self.stride      = stride
        self.max_persons = max_persons

        self._buffer: deque[PoseResult] = deque(maxlen=window_size)
        self._frame_count: int   = 0
        self._last_output_frame: int = -stride  # triggers on first full window

    # ── Public API ────────────────────────────────────────────────────────────

    def add_frame(self, pose_result: PoseResult) -> bool:
        """Add one frame to the buffer.

        Returns True when an action-recognition inference should be triggered
        (window is full AND the stride step has been reached).
        """
        self._buffer.append(pose_result)
        self._frame_count += 1

        if len(self._buffer) >= self.window_size:
            if (self._frame_count - self._last_output_frame) >= self.stride:
                return True
        return False

    def get_pose_results(
        self,
    ) -> Optional[tuple[list[dict], tuple[int, int]]]:
        """Convert current window to MMAction2 ``inference_skeleton`` format.

        Returns (pose_results, img_shape) where pose_results is a list of
        dicts with keys ``keypoints`` (M, 17, 2) and ``keypoint_scores``
        (M, 17), or None if the buffer is empty.
        """
        if len(self._buffer) == 0:
            return None

        self._last_output_frame = self._frame_count

        frames    = list(self._buffer)
        img_shape = frames[0].img_shape
        pose_results = []

        for frame_data in frames:
            n_persons = frame_data.keypoints.shape[0]

            if n_persons == 0:
                kpts   = np.zeros((1, 17, 2), dtype=np.float32)
                scores = np.zeros((1, 17),    dtype=np.float32)
            else:
                if n_persons > self.max_persons:
                    top_idx = np.argsort(frame_data.box_scores)[
                        -self.max_persons:][::-1]
                    kpts   = frame_data.keypoints[top_idx]
                    scores = frame_data.keypoint_scores[top_idx]
                else:
                    kpts   = frame_data.keypoints
                    scores = frame_data.keypoint_scores

            pose_results.append({
                "keypoints":       kpts.astype(np.float32),
                "keypoint_scores": scores.astype(np.float32),
            })

        return pose_results, img_shape

    def get_raw_frames(self) -> List[PoseResult]:
        """
        Return a snapshot of the current window as a list of PoseResult.

        This is the **only addition** compared to the original module.
        Called by GaitFeatureExtractor to access the same temporal window
        without duplicating buffer state or coupling the two branches.

        Note: ``get_pose_results()`` must have been called first (to update
        ``_last_output_frame``); this method does NOT update any internal state.
        """
        return list(self._buffer)

    def reset(self) -> None:
        """Clear buffer and reset counters."""
        self._buffer.clear()
        self._frame_count = 0
        self._last_output_frame = -self.stride

    @property
    def is_ready(self) -> bool:
        """True when the window holds enough frames for inference."""
        return len(self._buffer) >= self.window_size

    @property
    def buffer_size(self) -> int:
        """Current number of frames in the buffer."""
        return len(self._buffer)

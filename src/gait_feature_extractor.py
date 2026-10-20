"""Gait Feature Extractor
========================
Converts a sliding window of YOLO COCO-17 keypoint frames (PoseResult objects)
into the 15-dimensional gait feature vectors consumed by the HMM gait analyser.

COCO-17 keypoint index reference:
  0: nose        1: left_eye     2: right_eye    3: left_ear     4: right_ear
  5: L_shoulder  6: R_shoulder   7: L_elbow      8: R_elbow      9: L_wrist
 10: R_wrist    11: L_hip       12: R_hip       13: L_knee      14: R_knee
 15: L_ankle    16: R_ankle

Feature vector layout (15 features, matches hmm_gait.FEATURE_NAMES):
  0  stride_length          – pixel distance between successive same-side ankles
  1  stride_time            – frames between successive same-side ankle contacts
  2  cadence                – steps per second (fps-normalised)
  3  swing_ratio            – fraction of stride in swing phase
  4  stance_ratio           – fraction of stride in stance phase
  5  double_support_time    – overlap fraction where both feet are planted
  6  step_width             – lateral distance between left/right ankles (pixels)
  7  trunk_sway_ap          – anterior-posterior trunk angle variance (deg)
  8  trunk_sway_ml          – medio-lateral trunk angle variance (deg)
  9  heel_strike_angle      – ankle-knee-hip angle at contact event (deg)
 10  toe_off_angle          – ankle-knee-hip angle at lift-off event (deg)
 11  velocity               – centre-of-mass displacement per frame (pixels)
 12  symmetry_index         – |L_stride - R_stride| / mean_stride  (0=perfect)
 13  variability_index      – coefficient of variation of stride lengths
 14  smoothness             – mean jerk magnitude of hip midpoint trajectory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .pose_estimator import PoseResult
from .hmm_gait import GaitFeatureVector, FEATURE_NAMES

# ── COCO indices ──────────────────────────────────────────────────────────────
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP,      _R_HIP      = 11, 12
_L_KNEE,     _R_KNEE     = 13, 14
_L_ANKLE,    _R_ANKLE    = 15, 16

# Minimum keypoint confidence to treat a joint as visible
_KPT_CONF_THRESHOLD = 0.25
# Minimum fraction of frames where both ankles are visible to attempt extraction
_MIN_VISIBLE_FRACTION = 0.40
# Assumed FPS when not otherwise known (used for cadence normalisation)
_DEFAULT_FPS = 30.0


def _visible(scores: np.ndarray, idx: int) -> bool:
    """Return True if keypoint idx has confidence above threshold."""
    return float(scores[idx]) >= _KPT_CONF_THRESHOLD


def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by segments b→a and b→c, in degrees."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _lateral_angle(left_pt: np.ndarray, right_pt: np.ndarray) -> float:
    """Lateral tilt angle (degrees) of the vector from right→left point."""
    delta = left_pt - right_pt
    return float(np.degrees(np.arctan2(delta[1], delta[0] + 1e-8)))


# ── Per-frame joint accessor ──────────────────────────────────────────────────

def _best_person_idx(pose_result: PoseResult) -> Optional[int]:
    """
    Return the index of the person with the highest mean keypoint confidence,
    or None if no persons detected.
    """
    if pose_result.keypoints.shape[0] == 0:
        return None
    mean_conf = pose_result.keypoint_scores.mean(axis=1)  # (N,)
    return int(np.argmax(mean_conf))


# ── Core extractor ────────────────────────────────────────────────────────────

class GaitFeatureExtractor:
    """
    Converts a window of PoseResult frames into a single GaitFeatureVector.

    Intended to be called once per SkeletonConverter window trigger, receiving
    the same list[PoseResult] that SkeletonConverter holds in its buffer.

    Args:
        fps: Frames per second of the source video/camera. Used to convert
             frame-count measurements into time-based units.
        img_width: Width of the source frame in pixels. Used to normalise
                   pixel-distance features (optional; set to 1 to keep raw px).
        img_height: Height of the source frame in pixels.
    """

    def __init__(
        self,
        fps: float = _DEFAULT_FPS,
        img_width: int = 640,
        img_height: int = 480,
    ) -> None:
        self.fps = fps
        self.img_width = img_width
        self.img_height = img_height
        # Running cycle index across calls (for GaitFeatureVector.cycle_index)
        self._cycle_index: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        frames: List[PoseResult],
        timestamp: float = 0.0,
    ) -> Optional[GaitFeatureVector]:
        """
        Extract one GaitFeatureVector from a window of PoseResult frames.

        Returns None if the window does not contain enough visible keypoints
        to compute reliable features (e.g. person not in view).

        Args:
            frames: Ordered list of PoseResult, one per frame in the window.
            timestamp: Wall-clock timestamp of the last frame (seconds).
        """
        if not frames:
            return None

        # ── 1. Extract per-frame ankle / hip / knee / shoulder positions ──
        T = len(frames)
        l_ankle   = np.full((T, 2), np.nan)
        r_ankle   = np.full((T, 2), np.nan)
        l_knee    = np.full((T, 2), np.nan)
        r_knee    = np.full((T, 2), np.nan)
        l_hip     = np.full((T, 2), np.nan)
        r_hip     = np.full((T, 2), np.nan)
        l_shoulder = np.full((T, 2), np.nan)
        r_shoulder = np.full((T, 2), np.nan)

        quality_scores = []

        for t, pose in enumerate(frames):
            pi = _best_person_idx(pose)
            if pi is None:
                quality_scores.append(0.0)
                continue

            kpts   = pose.keypoints[pi]      # (17, 2)
            scores = pose.keypoint_scores[pi]  # (17,)
            quality_scores.append(float(scores.mean()))

            def _get(idx: int) -> Optional[np.ndarray]:
                return kpts[idx].copy() if _visible(scores, idx) else None

            for arr, idx in [
                (l_ankle,    _L_ANKLE),    (r_ankle,    _R_ANKLE),
                (l_knee,     _L_KNEE),     (r_knee,     _R_KNEE),
                (l_hip,      _L_HIP),      (r_hip,      _R_HIP),
                (l_shoulder, _L_SHOULDER), (r_shoulder, _R_SHOULDER),
            ]:
                pt = _get(idx)
                if pt is not None:
                    arr[t] = pt

        # Check visibility coverage
        l_vis = np.sum(~np.isnan(l_ankle[:, 0])) / T
        r_vis = np.sum(~np.isnan(r_ankle[:, 0])) / T
        overall_quality = float(np.nanmean(quality_scores)) if quality_scores else 0.0

        if max(l_vis, r_vis) < _MIN_VISIBLE_FRACTION:
            # Not enough ankle data — return low-confidence zeroed vector
            return self._fallback_vector(timestamp, overall_quality)

        # ── 2. Compute features ───────────────────────────────────────────
        features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

        # Normalisation divisors so pixel distances are scale-invariant
        # (expressed as fraction of frame diagonal)
        diag = float(np.sqrt(self.img_width**2 + self.img_height**2)) + 1e-8

        # --- Stride / cadence features (0-5) --------------------------------
        # Use hip midpoint trajectory to detect gait cycles via zero-crossings
        hip_mid = self._midpoint_series(l_hip, r_hip)       # (T, 2)
        ankle_mid = self._midpoint_series(l_ankle, r_ankle)  # (T, 2)

        # Stride length: mean distance between consecutive positions of the
        # same foot over the window (use whichever ankle is more visible)
        l_stride_lengths = self._consecutive_distances(l_ankle)
        r_stride_lengths = self._consecutive_distances(r_ankle)

        all_strides = np.concatenate([l_stride_lengths, r_stride_lengths])
        all_strides = all_strides[all_strides > 0]

        mean_stride  = float(np.nanmean(all_strides)) / diag if len(all_strides) else 0.0
        stride_var   = float(np.nanstd(all_strides))  / diag if len(all_strides) else 0.0

        # Stride time: mean frame-gap between steps (converted to seconds)
        step_times = self._detect_step_times(hip_mid)
        mean_stride_time = (float(np.mean(step_times)) / self.fps
                            if len(step_times) else float(T) / self.fps)

        # Cadence in steps/second
        cadence = (len(step_times) / (T / self.fps)) if T > 0 else 0.0

        # Swing / stance ratio — estimated from ankle height oscillation
        swing_r, stance_r, double_support = self._swing_stance_ratios(
            l_ankle, r_ankle, self.img_height)

        # Step width — lateral distance between ankle midpoints
        ankle_lateral = self._lateral_distances(l_ankle, r_ankle)
        step_width = float(np.nanmean(ankle_lateral)) / diag if len(ankle_lateral) else 0.0

        features[0] = mean_stride
        features[1] = mean_stride_time
        features[2] = cadence
        features[3] = swing_r
        features[4] = stance_r
        features[5] = double_support
        features[6] = step_width

        # --- Trunk sway (7-8) -----------------------------------------------
        trunk_ap_var, trunk_ml_var = self._trunk_sway(l_shoulder, r_shoulder,
                                                       l_hip, r_hip)
        features[7] = trunk_ap_var
        features[8] = trunk_ml_var

        # --- Joint angles (9-10) --------------------------------------------
        hs_angle, to_angle = self._mean_joint_angles(
            l_ankle, l_knee, l_hip, r_ankle, r_knee, r_hip)
        features[9]  = hs_angle
        features[10] = to_angle

        # --- Velocity (11) --------------------------------------------------
        velocity = self._com_velocity(hip_mid) / diag
        features[11] = velocity

        # --- Symmetry index (12) --------------------------------------------
        sym_index = self._symmetry_index(l_stride_lengths, r_stride_lengths)
        features[12] = sym_index

        # --- Variability index (13) -----------------------------------------
        features[13] = (stride_var / (mean_stride + 1e-8)
                        if mean_stride > 0 else 0.0)

        # --- Smoothness / jerk (14) -----------------------------------------
        features[14] = self._jerk_metric(hip_mid) / (diag + 1e-8)

        # Clip obviously non-physical values
        features = np.clip(features, 0.0, 1e4)

        self._cycle_index += 1
        return GaitFeatureVector(
            features=features,
            timestamp=timestamp,
            cycle_index=self._cycle_index,
            quality_score=min(1.0, overall_quality * max(l_vis, r_vis)),
        )

    def reset(self) -> None:
        """Reset internal cycle counter."""
        self._cycle_index = 0

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fallback_vector(self, timestamp: float, quality: float) -> GaitFeatureVector:
        """Return a zeroed feature vector flagged as low quality."""
        self._cycle_index += 1
        return GaitFeatureVector(
            features=np.zeros(len(FEATURE_NAMES), dtype=np.float32),
            timestamp=timestamp,
            cycle_index=self._cycle_index,
            quality_score=max(0.0, quality * 0.3),
        )

    @staticmethod
    def _midpoint_series(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compute frame-wise midpoint between two joint series, NaN-safe."""
        mid = np.full_like(left, np.nan)
        for t in range(left.shape[0]):
            l_ok = not np.isnan(left[t, 0])
            r_ok = not np.isnan(right[t, 0])
            if l_ok and r_ok:
                mid[t] = (left[t] + right[t]) / 2.0
            elif l_ok:
                mid[t] = left[t]
            elif r_ok:
                mid[t] = right[t]
        return mid  # (T, 2)

    @staticmethod
    def _consecutive_distances(joint: np.ndarray) -> np.ndarray:
        """Frame-to-frame Euclidean displacement of a joint series."""
        dists = []
        prev = None
        for t in range(joint.shape[0]):
            if np.isnan(joint[t, 0]):
                prev = None
                continue
            if prev is not None:
                dists.append(float(np.linalg.norm(joint[t] - prev)))
            prev = joint[t]
        return np.array(dists, dtype=np.float32)

    @staticmethod
    def _detect_step_times(hip_mid: np.ndarray) -> np.ndarray:
        """
        Estimate inter-step intervals from hip medio-lateral oscillation.
        Zero-crossing detection on the x-axis of hip midpoint.
        """
        x = hip_mid[:, 0]
        valid = ~np.isnan(x)
        if valid.sum() < 4:
            return np.array([])
        # Fill nans with linear interpolation
        xf = x.copy()
        nans = np.isnan(xf)
        if nans.any():
            xf[nans] = np.interp(np.where(nans)[0],
                                  np.where(~nans)[0],
                                  xf[~nans])
        # Detrend
        xf = xf - np.nanmean(xf)
        # Zero crossings
        sign_changes = np.where(np.diff(np.sign(xf)))[0]
        if len(sign_changes) < 2:
            return np.array([])
        intervals = np.diff(sign_changes).astype(np.float32)
        return intervals

    @staticmethod
    def _swing_stance_ratios(
        l_ankle: np.ndarray,
        r_ankle: np.ndarray,
        img_height: int,
    ) -> Tuple[float, float, float]:
        """
        Estimate swing ratio, stance ratio, and double-support fraction
        from vertical ankle displacement.

        In image coordinates, 'high' y = lower in the frame. We infer stance
        when the ankle is near its maximum y (near ground), swing when moving.
        """
        T = l_ankle.shape[0]
        if T < 4:
            return 0.40, 0.60, 0.22

        def _stance_frames(ankle: np.ndarray) -> np.ndarray:
            """Return boolean mask where ankle appears stationary (stance)."""
            y = ankle[:, 1]
            valid = ~np.isnan(y)
            if valid.sum() < 4:
                return np.zeros(T, dtype=bool)
            yf = np.where(valid, y, np.nanmean(y))
            # Smooth
            kernel = np.ones(3) / 3
            ys = np.convolve(yf, kernel, mode="same")
            # Stance = near maximum y value (foot on ground)
            thresh = np.nanpercentile(ys[valid], 60)
            return ys >= thresh

        l_stance = _stance_frames(l_ankle)
        r_stance = _stance_frames(r_ankle)

        l_swing = ~l_stance
        r_swing = ~r_stance
        double_support_mask = l_stance & r_stance

        swing_frac = float((l_swing | r_swing).mean())
        stance_frac = 1.0 - swing_frac
        double_frac = float(double_support_mask.mean())

        return (
            float(np.clip(swing_frac, 0.0, 1.0)),
            float(np.clip(stance_frac, 0.0, 1.0)),
            float(np.clip(double_frac, 0.0, 1.0)),
        )

    @staticmethod
    def _lateral_distances(
        l_ankle: np.ndarray, r_ankle: np.ndarray
    ) -> np.ndarray:
        """Frame-wise lateral (x-axis) distance between ankles."""
        dists = []
        for t in range(l_ankle.shape[0]):
            if not (np.isnan(l_ankle[t, 0]) or np.isnan(r_ankle[t, 0])):
                dists.append(abs(float(l_ankle[t, 0] - r_ankle[t, 0])))
        return np.array(dists, dtype=np.float32)

    @staticmethod
    def _trunk_sway(
        l_shoulder: np.ndarray,
        r_shoulder: np.ndarray,
        l_hip: np.ndarray,
        r_hip: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute AP and ML trunk sway as angle variance (degrees).
        AP sway = variance of forward tilt angle of torso midline.
        ML sway = variance of lateral tilt angle of shoulder line.
        """
        T = l_shoulder.shape[0]
        ap_angles = []
        ml_angles = []

        for t in range(T):
            sh_ok  = not (np.isnan(l_shoulder[t, 0]) or np.isnan(r_shoulder[t, 0]))
            hip_ok = not (np.isnan(l_hip[t, 0])     or np.isnan(r_hip[t, 0]))
            if sh_ok and hip_ok:
                sh_mid  = (l_shoulder[t] + r_shoulder[t]) / 2.0
                hip_mid = (l_hip[t]     + r_hip[t])     / 2.0
                torso   = sh_mid - hip_mid
                # AP: angle of torso from vertical (y-axis in image)
                ap = float(np.degrees(np.arctan2(torso[0], torso[1] + 1e-8)))
                ap_angles.append(ap)
                # ML: angle of shoulder line from horizontal
                sh_vec  = r_shoulder[t] - l_shoulder[t]
                ml = float(np.degrees(np.arctan2(sh_vec[1], sh_vec[0] + 1e-8)))
                ml_angles.append(ml)

        ap_var = float(np.std(ap_angles)) if len(ap_angles) > 1 else 2.0
        ml_var = float(np.std(ml_angles)) if len(ml_angles) > 1 else 1.5
        return ap_var, ml_var

    @staticmethod
    def _mean_joint_angles(
        l_ankle: np.ndarray, l_knee: np.ndarray, l_hip: np.ndarray,
        r_ankle: np.ndarray, r_knee: np.ndarray, r_hip: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute mean ankle-knee-hip angle across frames as proxies for
        heel-strike angle and toe-off angle.
        Returns (mean_angle_lower_quartile, mean_angle_upper_quartile).
        """
        angles = []
        for t in range(l_ankle.shape[0]):
            for ankle, knee, hip in [
                (l_ankle[t], l_knee[t], l_hip[t]),
                (r_ankle[t], r_knee[t], r_hip[t]),
            ]:
                if not (np.isnan(ankle[0]) or np.isnan(knee[0]) or np.isnan(hip[0])):
                    angles.append(_angle_3pts(ankle, knee, hip))
        if len(angles) < 4:
            return 5.0, 55.0
        angles = np.array(angles)
        heel_strike = float(np.percentile(angles, 25))
        toe_off     = float(np.percentile(angles, 75))
        return heel_strike, toe_off

    @staticmethod
    def _com_velocity(hip_mid: np.ndarray) -> float:
        """
        Mean CoM velocity = mean frame-to-frame displacement of hip midpoint.
        """
        dists = []
        for t in range(1, hip_mid.shape[0]):
            if not (np.isnan(hip_mid[t, 0]) or np.isnan(hip_mid[t-1, 0])):
                dists.append(float(np.linalg.norm(hip_mid[t] - hip_mid[t-1])))
        return float(np.mean(dists)) if dists else 0.0

    @staticmethod
    def _symmetry_index(
        l_strides: np.ndarray, r_strides: np.ndarray
    ) -> float:
        """
        Symmetry index: |mean_L - mean_R| / ((mean_L + mean_R) / 2).
        0 = perfectly symmetric, higher = more asymmetric.
        """
        if len(l_strides) == 0 or len(r_strides) == 0:
            return 0.05
        ml = float(np.nanmean(l_strides))
        mr = float(np.nanmean(r_strides))
        denom = (ml + mr) / 2.0
        if denom < 1e-8:
            return 0.0
        return float(abs(ml - mr) / denom)

    @staticmethod
    def _jerk_metric(hip_mid: np.ndarray) -> float:
        """
        Mean jerk of CoM trajectory (third derivative of position).
        Higher = rougher, less smooth gait.
        """
        valid_idx = np.where(~np.isnan(hip_mid[:, 0]))[0]
        if len(valid_idx) < 4:
            return 2.5
        pts = hip_mid[valid_idx]
        vel   = np.diff(pts, axis=0)
        acc   = np.diff(vel, axis=0)
        jerk  = np.diff(acc, axis=0)
        return float(np.mean(np.linalg.norm(jerk, axis=1)))

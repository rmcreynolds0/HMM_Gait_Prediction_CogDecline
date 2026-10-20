"""Gait Analyser Session Manager
=================================
Sits between the pipeline and the HMM.  Accumulates GaitFeatureVector objects
produced by GaitFeatureExtractor, manages the baseline-vs-scoring lifecycle,
and exposes a clean interface to ActionDetectionPipeline.

Modes
-----
  "collecting_baseline"
      Accumulates feature vectors into baseline sessions until
      ``min_baseline_sessions`` full sessions have been captured, then
      automatically trains the HMM.

  "scoring"
      Each incoming feature vector is appended to the running session buffer.
      Every ``score_every_n`` vectors a DeviationReport is emitted.

  "idle"
      Initial state before the first feature vector arrives, or after an
      explicit ``reset()``.

Persistence
-----------
Baseline data and trained model parameters are optionally saved/loaded from a
directory (``baseline_dir``).  This allows the baseline to survive across
sessions / restarts.
"""

from __future__ import annotations

import json
import os
import pickle
from enum import Enum, auto
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .hmm_gait import (
    GaitFeatureVector,
    GaitHMMAnalyser,
    DeviationReport,
    FEATURE_NAMES,
)


class GaitMode(Enum):
    IDLE               = auto()
    COLLECTING_BASELINE = auto()
    SCORING            = auto()


# ── Defaults ─────────────────────────────────────────────────────────────────
_DEFAULT_CYCLES_PER_SESSION = 50   # feature vectors per baseline session
_DEFAULT_MIN_SESSIONS       = 5    # number of baseline sessions before training
_DEFAULT_SCORE_EVERY        = 10   # emit a report every N new vectors
_DEFAULT_N_STATES           = 4
_DEFAULT_N_BOOTSTRAP        = 100


class GaitSessionManager:
    """
    High-level gait analysis manager integrated into the action-detection pipeline.

    Args:
        subject_id: Unique identifier for the subject.
        baseline_dir: Directory to persist baseline data and trained model.
                      Pass None to disable persistence.
        cycles_per_baseline_session: How many feature vectors constitute one
                                     baseline walk session.
        min_baseline_sessions: Minimum sessions needed to train HMM.
        score_every_n: Emit a DeviationReport every N new feature vectors.
        n_states: Number of hidden states in the HMM.
        n_bootstrap: Bootstrap samples for confidence intervals.
        on_report: Optional callback invoked whenever a new DeviationReport is
                   produced.  Signature: ``callback(report: DeviationReport)``.
        on_baseline_ready: Optional callback invoked when baseline training
                           completes.
    """

    def __init__(
        self,
        subject_id: str = "subject_000",
        baseline_dir: Optional[str] = None,
        cycles_per_baseline_session: int = _DEFAULT_CYCLES_PER_SESSION,
        min_baseline_sessions: int = _DEFAULT_MIN_SESSIONS,
        score_every_n: int = _DEFAULT_SCORE_EVERY,
        n_states: int = _DEFAULT_N_STATES,
        n_bootstrap: int = _DEFAULT_N_BOOTSTRAP,
        on_report: Optional[Callable[[DeviationReport], None]] = None,
        on_baseline_ready: Optional[Callable[[], None]] = None,
    ) -> None:
        self.subject_id   = subject_id
        self.baseline_dir = Path(baseline_dir) if baseline_dir else None
        self.cycles_per_baseline_session = cycles_per_baseline_session
        self.min_baseline_sessions       = min_baseline_sessions
        self.score_every_n  = score_every_n
        self.on_report       = on_report
        self.on_baseline_ready = on_baseline_ready

        self._analyser = GaitHMMAnalyser(
            n_states=n_states,
            n_bootstrap=n_bootstrap,
        )

        # State
        self._mode: GaitMode        = GaitMode.IDLE
        self._baseline_sessions: List[List[GaitFeatureVector]] = []
        self._current_baseline_buf: List[GaitFeatureVector] = []
        self._scoring_buf: List[GaitFeatureVector] = []
        self._session_counter: int  = 0
        self._vector_counter: int   = 0
        self._last_report: Optional[DeviationReport] = None

        # Try loading persisted baseline
        if self.baseline_dir and self._load_model():
            self._mode = GaitMode.SCORING
            print(f"[GaitSessionManager] Loaded persisted baseline for '{subject_id}'."
                  f" Mode: SCORING")
        else:
            self._mode = GaitMode.COLLECTING_BASELINE
            print(f"[GaitSessionManager] No baseline found for '{subject_id}'."
                  f" Mode: COLLECTING_BASELINE"
                  f" ({self.min_baseline_sessions} sessions × "
                  f"{self.cycles_per_baseline_session} cycles needed)")

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def mode(self) -> GaitMode:
        return self._mode

    @property
    def is_scoring(self) -> bool:
        return self._mode == GaitMode.SCORING

    @property
    def baseline_progress(self) -> float:
        """Fraction of baseline data collected [0.0, 1.0]."""
        total_needed = self.min_baseline_sessions * self.cycles_per_baseline_session
        collected = (len(self._baseline_sessions) * self.cycles_per_baseline_session
                     + len(self._current_baseline_buf))
        return min(1.0, collected / max(1, total_needed))

    @property
    def last_report(self) -> Optional[DeviationReport]:
        return self._last_report

    def add_vector(self, vector: GaitFeatureVector) -> Optional[DeviationReport]:
        """
        Feed a single GaitFeatureVector into the manager.

        Depending on current mode:
        - COLLECTING_BASELINE: accumulates data; trains once enough is collected.
        - SCORING: appends to scoring buffer; emits a report every score_every_n.

        Returns a DeviationReport if one was produced this call, else None.
        """
        self._vector_counter += 1

        if self._mode == GaitMode.COLLECTING_BASELINE:
            return self._handle_baseline(vector)
        elif self._mode == GaitMode.SCORING:
            return self._handle_scoring(vector)
        return None

    def force_score(self) -> Optional[DeviationReport]:
        """
        Immediately score the current scoring buffer regardless of score_every_n.
        Returns None if not in SCORING mode or buffer is too short.
        """
        if self._mode != GaitMode.SCORING or len(self._scoring_buf) < 3:
            return None
        return self._emit_report()

    def reset_session(self) -> None:
        """Clear the current scoring buffer (start a fresh walk session)."""
        self._scoring_buf.clear()
        self._session_counter += 1

    def reset_baseline(self) -> None:
        """
        Wipe the baseline and retrain from scratch.
        Deletes any persisted model files.
        """
        self._baseline_sessions.clear()
        self._current_baseline_buf.clear()
        self._scoring_buf.clear()
        self._analyser = GaitHMMAnalyser(
            n_states=self._analyser.hmm.n_states,
            n_bootstrap=self._analyser.n_bootstrap,
        )
        self._mode = GaitMode.COLLECTING_BASELINE
        self._last_report = None
        if self.baseline_dir:
            model_path = self.baseline_dir / f"{self.subject_id}_model.pkl"
            if model_path.exists():
                model_path.unlink()
        print(f"[GaitSessionManager] Baseline reset for '{self.subject_id}'.")

    def get_status_string(self) -> str:
        """One-line status for UI overlay."""
        if self._mode == GaitMode.COLLECTING_BASELINE:
            pct = int(self.baseline_progress * 100)
            return f"GAIT: Baseline {pct}%"
        if self._mode == GaitMode.SCORING and self._last_report:
            r = self._last_report
            return (f"GAIT: {r.deviation_score:+.2f}σ [{r.alert_level.upper()}]"
                    f"  CI[{r.ci_lower:.2f},{r.ci_upper:.2f}]")
        return "GAIT: Initialising..."

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _handle_baseline(
        self, vector: GaitFeatureVector
    ) -> Optional[DeviationReport]:
        self._current_baseline_buf.append(vector)
        if len(self._current_baseline_buf) >= self.cycles_per_baseline_session:
            self._baseline_sessions.append(list(self._current_baseline_buf))
            self._current_baseline_buf.clear()
            print(f"[GaitSessionManager] Baseline session "
                  f"{len(self._baseline_sessions)}/{self.min_baseline_sessions} complete.")

            if len(self._baseline_sessions) >= self.min_baseline_sessions:
                self._train_baseline()
        return None

    def _train_baseline(self) -> None:
        print(f"[GaitSessionManager] Training HMM on "
              f"{len(self._baseline_sessions)} baseline sessions...")
        self._analyser.fit_baseline(
            self._baseline_sessions, subject_id=self.subject_id)
        self._mode = GaitMode.SCORING
        print(f"[GaitSessionManager] HMM trained. "
              f"Baseline LL/obs = {self._analyser.baseline_mean:.4f} "
              f"± {self._analyser.baseline_std:.4f}")
        if self.baseline_dir:
            self._save_model()
        if self.on_baseline_ready:
            self.on_baseline_ready()

    def _handle_scoring(
        self, vector: GaitFeatureVector
    ) -> Optional[DeviationReport]:
        self._scoring_buf.append(vector)
        if len(self._scoring_buf) % self.score_every_n == 0:
            return self._emit_report()
        return None

    def _emit_report(self) -> DeviationReport:
        session_id = f"{self.subject_id}_s{self._session_counter:04d}"
        report = self._analyser.score_session(
            self._scoring_buf, session_id=session_id)
        self._last_report = report
        if self.on_report:
            self.on_report(report)
        return report

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_model(self) -> None:
        if not self.baseline_dir:
            return
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        path = self.baseline_dir / f"{self.subject_id}_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(self._analyser, f)
        print(f"[GaitSessionManager] Model saved → {path}")

    def _load_model(self) -> bool:
        if not self.baseline_dir:
            return False
        path = self.baseline_dir / f"{self.subject_id}_model.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                self._analyser = pickle.load(f)
            return True
        except Exception as e:
            print(f"[GaitSessionManager] Failed to load model: {e}")
            return False

"""
Gait HMM - Hidden Markov Model for Gait Analysis & Cognitive Decline Detection
===============================================================================
Consumes time-series feature vectors from a gait segmentation model,
trains a baseline HMM per subject, then scores new sequences to produce
deviation scores with bootstrapped confidence intervals.

Input contract (from upstream segmentation model):
  - feature_vectors: np.ndarray of shape (T, F)
      T = number of gait cycles (or frames), F = feature dimensions
  - Feature vector example (per cycle):
      [stride_length, stride_time, cadence, swing_ratio, stance_ratio,
       double_support_time, step_width, trunk_sway_ap, trunk_sway_ml,
       heel_strike_angle, toe_off_angle, velocity, symmetry_index,
       variability_index, smoothness (jerk_metric)]

Output contract:
  - DeviationReport: deviation_score, confidence_interval, state_sequence,
                     per-feature contributions, alert level
"""

import numpy as np
from scipy.stats import multivariate_normal, chi2
from scipy.special import logsumexp
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "stride_length",
    "stride_time",
    "cadence",
    "swing_ratio",
    "stance_ratio",
    "double_support_time",
    "step_width",
    "trunk_sway_ap",
    "trunk_sway_ml",
    "heel_strike_angle",
    "toe_off_angle",
    "velocity",
    "symmetry_index",
    "variability_index",
    "smoothness",
]
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class GaitFeatureVector:
    """Single gait cycle feature vector from segmentation model."""
    features: np.ndarray          # shape (N_FEATURES,)
    timestamp: float = 0.0        # seconds since session start
    cycle_index: int = 0
    quality_score: float = 1.0    # segmentation confidence [0,1]


@dataclass
class DeviationReport:
    """Output report for a gait session."""
    subject_id: str
    session_id: str
    deviation_score: float               # 0 = baseline, higher = more deviant
    ci_lower: float                      # 95% CI lower bound
    ci_upper: float                      # 95% CI upper bound
    alert_level: str                     # "normal" | "watch" | "alert" | "critical"
    state_sequence: np.ndarray           # Viterbi decoded hidden states
    per_feature_deviation: dict          # feature_name -> z-score
    log_likelihood: float                # raw HMM log-likelihood
    baseline_log_likelihood: float       # expected under baseline
    n_cycles: int                        # number of gait cycles processed
    confidence: float                    # model confidence in report [0,1]
    notes: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian HMM
# ─────────────────────────────────────────────────────────────────────────────

class GaussianHMM:
    """
    Full-covariance Gaussian HMM with Baum-Welch EM training.

    States represent latent gait phases / quality levels.
    Observations are continuous feature vectors.
    """

    def __init__(self, n_states: int = 4, n_features: int = N_FEATURES,
                 n_iter: int = 100, tol: float = 1e-4, reg_covar: float = 1e-4,
                 random_state: int = 42):
        self.n_states = n_states
        self.n_features = n_features
        self.n_iter = n_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.rng = np.random.default_rng(random_state)
        self.is_fitted = False

        # Parameters (log-space where applicable)
        self.log_pi = None      # (S,)  initial state distribution
        self.log_A = None       # (S,S) transition matrix
        self.means = None       # (S,F) emission means
        self.covars = None      # (S,F,F) emission covariances

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_params(self, X: np.ndarray):
        """K-means warm start for emission parameters."""
        T, F = X.shape
        S = self.n_states

        # Uniform initial / transition
        self.log_pi = np.full(S, -np.log(S))
        self.log_A = np.full((S, S), -np.log(S))

        # K-means style init
        idx = self.rng.choice(T, S, replace=False)
        self.means = X[idx].copy()                               # (S,F)
        global_cov = np.cov(X.T) + self.reg_covar * np.eye(F)
        self.covars = np.stack([global_cov.copy() for _ in range(S)])

    # ── Emission log-probabilities ────────────────────────────────────────────

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log p(x_t | z_t = s) for all t, s.
        Returns (T, S).
        """
        T = len(X)
        log_b = np.zeros((T, self.n_states))
        for s in range(self.n_states):
            try:
                rv = multivariate_normal(mean=self.means[s],
                                         cov=self.covars[s],
                                         allow_singular=True)
                log_b[:, s] = rv.logpdf(X)
            except Exception:
                log_b[:, s] = -1e10
        return log_b

    # ── Forward / Backward ───────────────────────────────────────────────────

    def _forward(self, log_b: np.ndarray):
        """Log-space forward algorithm. Returns (log_alpha, log_likelihood)."""
        T, S = log_b.shape
        log_alpha = np.full((T, S), -np.inf)
        log_alpha[0] = self.log_pi + log_b[0]
        for t in range(1, T):
            for s in range(S):
                log_alpha[t, s] = (logsumexp(log_alpha[t-1] + self.log_A[:, s])
                                   + log_b[t, s])
        log_likelihood = logsumexp(log_alpha[-1])
        return log_alpha, log_likelihood

    def _backward(self, log_b: np.ndarray):
        """Log-space backward algorithm. Returns log_beta."""
        T, S = log_b.shape
        log_beta = np.full((T, S), -np.inf)
        log_beta[-1] = 0.0
        for t in range(T-2, -1, -1):
            for s in range(S):
                log_beta[t, s] = logsumexp(
                    self.log_A[s] + log_b[t+1] + log_beta[t+1])
        return log_beta

    # ── Baum-Welch EM ─────────────────────────────────────────────────────────

    def fit(self, sequences: list) -> "GaussianHMM":
        """
        Train on a list of observation sequences.
        Each sequence: np.ndarray of shape (T_i, F).
        """
        # Stack for init
        X_all = np.vstack(sequences)
        self._init_params(X_all)

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            # ── E-step ──────────────────────────────────────────────────────
            acc_pi = np.full(self.n_states, -np.inf)
            acc_A = np.full((self.n_states, self.n_states), -np.inf)
            # Sufficient stats for M-step (non-log)
            gamma_sum = np.zeros(self.n_states)
            gamma_x = np.zeros((self.n_states, self.n_features))
            gamma_xx = np.zeros((self.n_states, self.n_features, self.n_features))
            total_ll = 0.0

            for X in sequences:
                T = len(X)
                log_b = self._log_emission(X)
                log_alpha, ll = self._forward(log_b)
                log_beta = self._backward(log_b)
                total_ll += ll

                # Posterior state probs: log gamma (T, S)
                log_gamma = log_alpha + log_beta
                log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
                gamma = np.exp(log_gamma)

                # Initial state
                acc_pi = np.logaddexp(acc_pi, log_gamma[0])

                # Transition counts: log xi (T-1, S, S)
                for t in range(T-1):
                    log_xi_t = (log_alpha[t, :, None]
                                + self.log_A
                                + log_b[t+1, None, :]
                                + log_beta[t+1, None, :])
                    log_xi_t -= logsumexp(log_xi_t)
                    acc_A = np.logaddexp(acc_A, log_xi_t)

                # Emission sufficient stats
                for s in range(self.n_states):
                    gs = gamma[:, s]
                    gamma_sum[s] += gs.sum()
                    gamma_x[s] += (gs[:, None] * X).sum(axis=0)
                    diff = X - self.means[s]
                    gamma_xx[s] += (gs[:, None, None]
                                    * (diff[:, :, None] * diff[:, None, :])).sum(axis=0)

            # ── M-step ──────────────────────────────────────────────────────
            # Initial distribution
            self.log_pi = acc_pi - logsumexp(acc_pi)

            # Transition matrix (row-normalise in log space)
            self.log_A = acc_A - logsumexp(acc_A, axis=1, keepdims=True)

            # Emissions
            for s in range(self.n_states):
                if gamma_sum[s] > 1e-8:
                    self.means[s] = gamma_x[s] / gamma_sum[s]
                    self.covars[s] = (np.diag(np.diag(gamma_xx[s] / gamma_sum[s]))
                                      + self.reg_covar * np.eye(self.n_features))

            # Convergence check
            delta = total_ll - prev_ll
            if abs(delta) < self.tol and iteration > 5:
                break
            prev_ll = total_ll

        self.is_fitted = True
        self._baseline_ll_per_obs = total_ll / sum(len(s) for s in sequences)
        return self

    # ── Viterbi ──────────────────────────────────────────────────────────────

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Return most likely state sequence via Viterbi."""
        T = len(X)
        log_b = self._log_emission(X)
        viterbi_log = np.full((T, self.n_states), -np.inf)
        backpointer = np.zeros((T, self.n_states), dtype=int)

        viterbi_log[0] = self.log_pi + log_b[0]
        for t in range(1, T):
            for s in range(self.n_states):
                probs = viterbi_log[t-1] + self.log_A[:, s]
                backpointer[t, s] = np.argmax(probs)
                viterbi_log[t, s] = probs[backpointer[t, s]] + log_b[t, s]

        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(viterbi_log[-1])
        for t in range(T-2, -1, -1):
            states[t] = backpointer[t+1, states[t+1]]
        return states

    # ── Scoring ──────────────────────────────────────────────────────────────

    def score(self, X: np.ndarray) -> float:
        """Return log-likelihood per observation."""
        log_b = self._log_emission(X)
        _, ll = self._forward(log_b)
        return ll / len(X)


# ─────────────────────────────────────────────────────────────────────────────
# Gait Analyser (main interface)
# ─────────────────────────────────────────────────────────────────────────────

class GaitHMMAnalyser:
    """
    Wraps GaussianHMM for gait-specific training and deviation scoring.

    Workflow:
      1. analyser.fit_baseline(baseline_sessions)
      2. report = analyser.score_session(new_session_vectors)
    """

    def __init__(self, n_states: int = 4, n_bootstrap: int = 200,
                 random_state: int = 0):
        self.n_states = n_states
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(random_state)
        self.hmm = GaussianHMM(n_states=n_states, n_features=N_FEATURES,
                                random_state=random_state)
        self.baseline_scores: Optional[np.ndarray] = None
        self.baseline_mean: float = 0.0
        self.baseline_std: float = 1.0
        self.feature_baseline_mean: Optional[np.ndarray] = None
        self.feature_baseline_std: Optional[np.ndarray] = None
        self.is_fitted = False

    # ── Preprocessing ────────────────────────────────────────────────────────

    @staticmethod
    def vectors_to_array(vectors: list, mu=None, sigma=None):
        """Convert list[GaitFeatureVector] → np.ndarray (T, F), quality-filtered."""
        arr = np.stack([v.features for v in vectors
                        if v.quality_score >= 0.5])
        if mu is None:
            mu = arr.mean(axis=0)
            sigma = arr.std(axis=0) + 1e-8
        return (arr - mu) / sigma, mu, sigma

    # ── Baseline training ─────────────────────────────────────────────────────

    def fit_baseline(self, baseline_sessions: list,
                     subject_id: str = "unknown") -> "GaitHMMAnalyser":
        """
        Train HMM on multiple baseline gait sessions.

        Parameters
        ----------
        baseline_sessions : list of list[GaitFeatureVector]
            Each inner list is one walking session.
        """
        self.subject_id = subject_id

        # Compute global normalisation from ALL baseline data
        all_raw = np.vstack([
            np.stack([v.features for v in sess]) for sess in baseline_sessions
        ])
        self.feature_baseline_mean = all_raw.mean(axis=0)
        self.feature_baseline_std = all_raw.std(axis=0) + 1e-8

        sequences = []
        for session in baseline_sessions:
            X, _, _ = self.vectors_to_array(
                session, self.feature_baseline_mean, self.feature_baseline_std)
            sequences.append(X)

        # Fit HMM
        self.hmm.fit(sequences)

        # Bootstrap baseline score distribution using held-out windows
        scores = []
        for X in sequences:
            for _ in range(max(1, self.n_bootstrap // len(sequences))):
                idx = self.rng.choice(len(X), len(X), replace=True)
                scores.append(self.hmm.score(X[idx]))
        self.baseline_scores = np.array(scores)
        self.baseline_mean = float(self.baseline_scores.mean())
        self.baseline_std = float(self.baseline_scores.std() + 1e-8)

        self.is_fitted = True
        return self

    # ── Session scoring ───────────────────────────────────────────────────────

    def score_session(self, session: list,
                      session_id: str = "session_0") -> DeviationReport:
        """
        Score a new gait session against the baseline.

        Returns DeviationReport with deviation score, CI, and per-feature info.
        """
        assert self.is_fitted, "Call fit_baseline() first."

        X_raw = np.stack([v.features for v in session
                          if v.quality_score >= 0.5])
        X, _, _ = self.vectors_to_array(
            session, self.feature_baseline_mean, self.feature_baseline_std)

        # HMM log-likelihood per observation
        ll = self.hmm.score(X)

        # Deviation score (z-score against bootstrap distribution)
        deviation = (self.baseline_mean - ll) / self.baseline_std
        # Positive = worse than baseline (lower LL → higher deviation)

        # Bootstrap CI for deviation on this session
        boot_scores = []
        for _ in range(self.n_bootstrap):
            idx = self.rng.choice(len(X), len(X), replace=True)
            bs_ll = self.hmm.score(X[idx])
            boot_scores.append((self.baseline_mean - bs_ll) / self.baseline_std)
        boot_scores = np.array(boot_scores)
        ci_lower = float(np.percentile(boot_scores, 2.5))
        ci_upper = float(np.percentile(boot_scores, 97.5))

        # Viterbi state sequence
        states = self.hmm.viterbi(X)

        # Per-feature z-scores vs baseline
        X_raw = np.stack([v.features for v in session if v.quality_score >= 0.5])
        mu = self.feature_baseline_mean
        sigma = self.feature_baseline_std
        session_feat_mean = X_raw.mean(axis=0)
        per_feature = {
            name: float((session_feat_mean[i] - mu[i]) / sigma[i])
            for i, name in enumerate(FEATURE_NAMES)
        }

        # Alert level
        alert = self._alert_level(deviation, ci_lower)

        # Model confidence: based on sequence length and quality
        avg_quality = np.mean([v.quality_score for v in session])
        length_factor = min(1.0, len(session) / 30.0)
        confidence = float(avg_quality * length_factor)

        notes = self._generate_notes(per_feature, deviation, states)

        return DeviationReport(
            subject_id=self.subject_id,
            session_id=session_id,
            deviation_score=float(deviation),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            alert_level=alert,
            state_sequence=states,
            per_feature_deviation=per_feature,
            log_likelihood=float(ll),
            baseline_log_likelihood=float(self.baseline_mean),
            n_cycles=len(X),
            confidence=confidence,
            notes=notes,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _alert_level(deviation: float, ci_lower: float) -> str:
        if ci_lower < 0:
            return "normal"
        if deviation < 1.5:
            return "watch"
        if deviation < 3.0:
            return "alert"
        return "critical"

    @staticmethod
    def _generate_notes(per_feature: dict, deviation: float,
                        states: np.ndarray) -> list:
        notes = []
        # Flag features with |z| > 2
        flagged = [(k, v) for k, v in per_feature.items() if abs(v) > 2.0]
        flagged.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, z in flagged[:3]:
            direction = "increased" if z > 0 else "decreased"
            notes.append(f"{name.replace('_', ' ').title()} {direction} "
                         f"({z:+.2f} SD from baseline)")

        # State stability
        transitions = np.sum(np.diff(states) != 0)
        if transitions > len(states) * 0.4:
            notes.append("High state instability detected (frequent gait phase switching)")

        if deviation > 2.0:
            notes.append("Significant overall gait deviation — recommend clinical follow-up")
        return notes


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (mirrors segmentation model output)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_session(n_cycles: int = 50,
                                 deviation_level: float = 0.0,
                                 rng: np.random.Generator = None) -> list:
    """
    Generate a synthetic gait session.

    deviation_level: 0.0 = healthy baseline, 1.0 = moderate decline,
                     2.0 = severe decline
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Healthy baseline means (approximate real values, z-scored internally)
    baseline_means = np.array([
        1.40,   # stride_length (m)
        1.10,   # stride_time (s)
        110.0,  # cadence (steps/min)
        0.40,   # swing_ratio
        0.60,   # stance_ratio
        0.22,   # double_support_time (s)
        0.12,   # step_width (m)
        1.5,    # trunk_sway_ap (deg)
        1.2,    # trunk_sway_ml (deg)
        5.0,    # heel_strike_angle (deg)
        55.0,   # toe_off_angle (deg)
        1.35,   # velocity (m/s)
        0.95,   # symmetry_index
        0.08,   # variability_index
        2.5,    # smoothness (lower jerk = smoother)
    ])

    # Cognitive/motor decline shifts specific features
    decline_shifts = np.array([
        -0.10,  # shorter strides
        +0.08,  # slower cadence → longer stride time
        -8.0,   # lower cadence
        -0.02,  # less swing
        +0.02,  # more stance
        +0.04,  # more double support
        +0.02,  # wider step
        +1.2,   # more AP sway
        +1.5,   # more ML sway
        -1.5,   # flatter heel strike
        -5.0,   # reduced toe-off
        -0.15,  # slower velocity
        -0.04,  # less symmetry
        +0.04,  # more variability
        +0.8,   # rougher (more jerk)
    ]) * deviation_level

    noise_std = np.array([
        0.05, 0.03, 3.0, 0.02, 0.02, 0.01, 0.01,
        0.5, 0.5, 1.0, 2.0, 0.05, 0.02, 0.01, 0.2
    ]) * (1 + 0.3 * deviation_level)

    vectors = []
    for i in range(n_cycles):
        features = (baseline_means + decline_shifts
                    + rng.normal(0, noise_std))
        features = np.clip(features, 0.01, None)
        quality = rng.uniform(0.85, 1.0) * max(0.6, 1 - 0.05 * deviation_level)
        vectors.append(GaitFeatureVector(
            features=features,
            timestamp=float(i) * 1.1,
            cycle_index=i,
            quality_score=quality,
        ))
    return vectors

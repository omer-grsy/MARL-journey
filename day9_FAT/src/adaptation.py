"""
Day 9 adaptation primitives — v4 (norm + direction signals).

FaultDetector v4 changes vs v3:

  Fix 7 — Direction (cosine) signal:
    Norm-only detection is structurally blind to norm-preserving Byzantine
    attacks (sign-flip, random replay, biased injection). The detector now
    accepts an optional `msg_h_mean` argument — per-agent mean embedding
    vector over the rollout — and computes a fourth signal:

      cos_drop[i] = max(0, fleet_baseline_cos − median_cos_to_others[i])

    where median_cos_to_others[i] = median over j≠i of cosine(mean_h_i, mean_h_j).
    Healthy fleet → cos_drop ≈ 0. Sign-flip → faulty has anti-correlated
    direction → cos_drop large.

  Fix 8 — Direction self-coherence signal:
    Random-direction attacks (per-step random h_i with healthy norm) cancel
    out under averaging: ||mean_h_random|| ≪ mean(||h_random||). Healthy and
    sign-flip Byzantine both have consistent per-step direction → ratio ≈ 1.
    Random Byzantine → ratio ≈ 0.

      coherence[i]      = ||mean_h_i|| / mean(||h_i||)   ∈ [0, 1]
      coherence_drop[i] = max(0, median(coherence) − coherence[i])

  Fault coverage map:
       fault              z_fleet  z_cv  cos_drop  coherence_drop
       fail_stop          ✓        —     ✓         (✓ as side-effect)
       byz_magnitude≥2    ✓        —     ✓         —
       byz_signflip       —        —     ✓         —
       byz_random         —        —     —         ✓
       intermittent       —        ✓     —         (partial)

  k_fleet default lowered to 0.7:
    Analytic floor: fail_stop has z_fleet = 1.0, byz_magnitude=2 has z_fleet
    = 1.0. Setting k_fleet ≥ 1.0 misses both. 0.7 keeps margin against
    healthy variance while catching mag-based faults.

  msg_h_mean=None keeps backward compatibility (norm-only, v3 behavior).
"""

from __future__ import annotations
from typing import Iterable, Optional

import numpy as np
import torch


class CurriculumScheduler:
    """
    Linear ramp in first linear_frac of training, then plateau-triggered
    adaptive bumps.
    """
    def __init__(self, total_updates, linear_frac=0.2, linear_max=0.2,
                 plateau_window=40, plateau_eps=0.02, bump_step=0.1,
                 intensity_cap=1.0):
        self.total_updates  = total_updates
        self.linear_end     = max(1, int(total_updates * linear_frac))
        self.linear_max     = linear_max
        self.plateau_window = plateau_window
        self.plateau_eps    = plateau_eps
        self.bump_step      = bump_step
        self.intensity_cap  = intensity_cap
        self.intensity      = 0.0
        self._reward_hist   = []
        self._last_bump     = -plateau_window

    def step(self, update_idx: int, recent_reward: float) -> float:
        if update_idx < self.linear_end:
            self.intensity = self.linear_max * (update_idx / max(1, self.linear_end))
            return self.intensity
        self._reward_hist.append(recent_reward)
        if len(self._reward_hist) > self.plateau_window:
            self._reward_hist = self._reward_hist[-self.plateau_window:]
        if ((update_idx - self._last_bump) >= self.plateau_window
                and len(self._reward_hist) == self.plateau_window):
            half = self.plateau_window // 2
            if (np.mean(self._reward_hist[half:]) - np.mean(self._reward_hist[:half])) < self.plateau_eps:
                self.intensity = min(self.intensity_cap, self.intensity + self.bump_step)
                self._last_bump = update_idx
                self._reward_hist = self._reward_hist[half:]
        return self.intensity


class FaultDetector:
    """
    Five-signal detector — z_self, z_cv, z_fleet (norm-based), cos_drop,
    coherence_drop (direction-based). Symmetric set/clear hysteresis.
    """

    _MIN_VAR = 1e-3

    def __init__(self, n_agents, k_threshold=3.0, k_cv=2.0, k_fleet=0.7,
                 k_cos_drop=0.35, k_coherence_drop=0.3,
                 ema_alpha=0.1, hysteresis_M=5,
                 set_persistence_K=3, warmup_updates=30):
        self.n            = n_agents
        self.k            = k_threshold
        self.k_cv         = k_cv
        self.k_fleet      = k_fleet
        self.k_cos        = k_cos_drop
        self.k_coherence  = k_coherence_drop
        self.alpha        = ema_alpha
        self.M            = hysteresis_M
        self.K_set        = set_persistence_K
        self.warmup       = warmup_updates

        self.pa_ema_norm  = np.zeros(n_agents)
        self.pa_ema_var   = np.full(n_agents, self._MIN_VAR)
        self.fleet_cv_ema = 0.0
        self.fleet_cv_var = self._MIN_VAR

        self._init        = False
        self.flags        = np.zeros(n_agents, dtype=bool)
        self.raw_streak   = np.zeros(n_agents, dtype=np.int32)
        self.clean_streak = np.zeros(n_agents, dtype=np.int32)
        self.n_updates    = 0
        self.tp = self.fp = self.tn = self.fn = 0

    def update(self, msg_norms_mean, msg_norms_std,
               msg_h_mean=None, true_faulty=None):
        """
        Parameters
        ----------
        msg_norms_mean : [N] per-agent mean of ||h_i|| over the rollout
        msg_norms_std  : [N] per-agent std  of ||h_i|| over the rollout
        msg_h_mean     : [N, H] per-agent mean embedding vector over the
                         rollout. If None, direction signals are disabled
                         (norm-only v3 behavior).
        true_faulty    : ground-truth faulty indices (TP/FP accounting)
        """
        self.n_updates += 1
        norms = np.asarray(msg_norms_mean, dtype=np.float64)
        stds  = np.asarray(msg_norms_std,  dtype=np.float64)
        cv    = stds / (norms + 1e-6)
        a     = self.alpha

        if not self._init:
            self.pa_ema_norm[:] = norms
            self.fleet_cv_ema   = float(np.median(cv))
            self._init = True
            return self.flags.copy()

        old_norm = self.pa_ema_norm.copy()
        old_var  = self.pa_ema_var.copy()

        self.pa_ema_var  = (1-a)*self.pa_ema_var  + a*(norms - self.pa_ema_norm)**2
        self.pa_ema_norm = (1-a)*self.pa_ema_norm + a*norms

        cur_cv = float(np.median(cv))
        self.fleet_cv_var = (1-a)*self.fleet_cv_var + a*(cur_cv - self.fleet_cv_ema)**2
        self.fleet_cv_ema = (1-a)*self.fleet_cv_ema + a*cur_cv

        if self.n_updates <= self.warmup:
            return self.flags.copy()

        # Signal 1: per-agent self-deviation (sudden temporal jumps).
        pa_std = np.sqrt(np.maximum(old_var, self._MIN_VAR)) + 1e-8
        z_self = np.abs(norms - old_norm) / pa_std

        # Signal 2: temporal CoV (intermittent fingerprint).
        cv_std = np.sqrt(max(self.fleet_cv_var, self._MIN_VAR)) + 1e-8
        z_cv   = np.maximum(cv - self.fleet_cv_ema, 0) / cv_std

        # Signal 3: relative norm deviation from fleet median.
        fleet_median = max(float(np.median(norms)), 0.1)
        z_fleet = np.abs(norms - fleet_median) / fleet_median

        # Signal 4: cosine-based direction inconsistency (sign-flip etc).
        # Signal 5: direction self-coherence ratio (random-replay etc).
        if msg_h_mean is not None:
            h_mean = np.asarray(msg_h_mean, dtype=np.float64)
            h_norms_vec = np.linalg.norm(h_mean, axis=1)              # (N,)
            h_unit = h_mean / (h_norms_vec[:, None] + 1e-8)           # (N, H)
            cos_mat = h_unit @ h_unit.T                               # (N, N)
            np.fill_diagonal(cos_mat, np.nan)
            with np.errstate(invalid="ignore"):
                median_cos = np.nanmedian(cos_mat, axis=1)            # (N,)
            fleet_baseline_cos = float(np.median(median_cos))
            cos_drop = np.maximum(fleet_baseline_cos - median_cos, 0.0)

            # coherence ∈ [0,1]: 1 = aligned per-step direction, 0 = random
            coherence = h_norms_vec / (norms + 1e-8)
            fleet_baseline_coh = float(np.median(coherence))
            coherence_drop = np.maximum(fleet_baseline_coh - coherence, 0.0)
        else:
            cos_drop = np.zeros(self.n)
            coherence_drop = np.zeros(self.n)

        raw = ((z_self > self.k) | (z_cv > self.k_cv)
               | (z_fleet > self.k_fleet) | (cos_drop > self.k_cos)
               | (coherence_drop > self.k_coherence))

        # Breakdown-point cap: score normalises each signal to its threshold,
        # so all terms are unitless "ratio above threshold" and comparable.
        max_f = max(1, self.n // 2)
        if int(raw.sum()) > max_f:
            score = (z_self / self.k + z_cv / self.k_cv
                     + z_fleet / self.k_fleet
                     + cos_drop / max(self.k_cos, 1e-6)
                     + coherence_drop / max(self.k_coherence, 1e-6))
            top_idx = np.argsort(score)[-max_f:]
            raw = np.zeros(self.n, dtype=bool)
            raw[top_idx] = True

        # Symmetric hysteresis: K_set consecutive triggers to SET, M to CLEAR.
        for i in range(self.n):
            if raw[i]:
                self.raw_streak[i]  += 1
                self.clean_streak[i] = 0
                if self.raw_streak[i] >= self.K_set:
                    self.flags[i] = True
            else:
                self.raw_streak[i]   = 0
                self.clean_streak[i] += 1
                if self.flags[i] and self.clean_streak[i] >= self.M:
                    self.flags[i] = False

        if true_faulty is not None:
            truth = set(int(x) for x in true_faulty)
            for i in range(self.n):
                fl = bool(self.flags[i])
                ft = i in truth
                if   ft and fl: self.tp += 1
                elif ft:        self.fn += 1
                elif fl:        self.fp += 1
                else:           self.tn += 1

        return self.flags.copy()

    def summary(self):
        pr  = self.tp / max(1, self.tp + self.fp)
        re  = self.tp / max(1, self.tp + self.fn)
        fpr = self.fp / max(1, self.fp + self.tn)
        f1  = 2 * pr * re / max(1e-8, pr + re)
        return dict(tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn,
                    precision=pr, recall=re, false_positive_rate=fpr, f1=f1,
                    samples=self.tp + self.fp + self.tn + self.fn)


class TopologyManager:
    """
    Edit adjacency at runtime given detector flag vector.
    Flagged agents have both incoming and outgoing edges zeroed.
    keep_self_loop ensures isolated agents still aggregate their own embedding.
    """
    def __init__(self, n_agents, keep_self_loop=True):
        self.n = n_agents
        self.keep_self_loop = keep_self_loop

    def reconfigure(self, base_adj, flags):
        adj = base_adj.clone()
        for i in range(self.n):
            if flags[i]:
                adj[i, :] = 0.0
                adj[:, i] = 0.0
        if self.keep_self_loop:
            adj = torch.maximum(adj, torch.eye(self.n, device=adj.device, dtype=adj.dtype))
        return adj / adj.sum(dim=1, keepdim=True).clamp(min=1.0)
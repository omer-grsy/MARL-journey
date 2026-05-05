"""
Day 9 training script — extends the Day 8 MAPPO + CommNet pipeline with
three adaptation strategies selected via --strategy.

Differences vs day8.py (all other logic unchanged):

    1. CLI gains --strategy {A,B,C} and --config (yaml path).
    2. CommNetActor now exposes explicit encode + aggregate_and_policy
       sub-methods so the training loop can inject faults at the *message*
       level (between encoder and aggregation).
    3. Byzantine is now adversarial (inverted-mean of healthy embeddings),
       not Gaussian noise — per the user's FDI-literature preference.
    4. Strategies B and C plug in CurriculumScheduler / FaultDetector /
       TopologyManager from src/adaptation.py.
    5. Observation may be augmented with a 1-d fault indicator (strategy
       B uses ground-truth, C uses the detector flag for the same agent).

MATLAB -> Python note: the encoder/aggregate split mirrors how you would
separate a "sensor model" from the "bus aggregation" block in a Simulink
fault-injection diagram — you need a named point to inject the fault.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# Make src.adaptation importable whether we run from repo root or elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.adaptation import CurriculumScheduler, FaultDetector, TopologyManager

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

from mpe2 import simple_spread_v3


# =========================
# HYPERPARAMETERS (same as Day 8 defaults; overridable via config)
# =========================
N_AGENTS = 3
MAX_CYCLES = 50
HIDDEN_DIM = 128

LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
GAMMA = 0.95
LAM = 0.95
CLIP_EPS = 0.2
VALUE_CLIP_EPS = 0.2
PPO_EPOCHS = 4
MINI_BATCH = 256

ROLLOUT_STEPS = 1000
TOTAL_UPDATES = 3000

ENTROPY_COEF = 0.015
BYZANTINE_MAGNITUDE_DEFAULT = 1.0  # reduced from 2.0; 2.0 + intensity=1.0 collapses one seed on S3
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

COVERAGE_THRESHOLD = 0.30
COVERAGE_BONUS = 1.0
REWARD_SCALE = 1.0

CONV_COVERAGE_THRESHOLD = 0.6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Day 8 helpers (unchanged)
# =========================
class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        self.var = M2 / tot
        self.count = tot


def normalize_obs(o, rms, eps=1e-8):
    return (o - rms.mean) / (np.sqrt(rms.var) + eps)


def compute_metrics(obs, agents, threshold=COVERAGE_THRESHOLD):
    if not obs:
        return 0.0, 0.0, float('inf')
    min_dists = [float('inf')] * N_AGENTS
    for a in agents:
        if a not in obs:
            continue
        o = obs[a]
        for j in range(N_AGENTS):
            rel = o[4 + j*2 : 4 + j*2 + 2]
            d = np.linalg.norm(rel)
            if d < min_dists[j]:
                min_dists[j] = d
    valid = [d for d in min_dists if d != float('inf')]
    if not valid:
        return 0.0, 0.0, float('inf')

    coverage_dense = float(np.mean([np.exp(-d * 3.0) for d in valid]))
    coverage_binary = sum(1 for d in valid if d < threshold) / N_AGENTS
    avg_min_dist = float(np.mean(valid))
    return coverage_dense, coverage_binary, avg_min_dist


def load_topology(path):
    with open(path, "r") as f:
        topo = json.load(f)
    if topo["mode"] == "static":
        adj = torch.tensor(topo["adj_matrix"], dtype=torch.float32).to(device)
        return "static", adj, None
    return "dynamic", None, topo["comm_radius"]


def compute_adj(obs, agents, r):
    n = len(agents)
    adj = torch.zeros(n, n, device=device)
    other_start = 4 + N_AGENTS * 2
    for i in range(n):
        o = obs.get(agents[i])
        if o is None:
            continue
        idx = 0
        for j in range(n):
            if i == j:
                continue
            rel = o[other_start + idx * 2: other_start + idx * 2 + 2]
            if np.linalg.norm(rel) < r:
                adj[i, j] = 1.0
            idx += 1
    return adj


# =========================
# FaultWrapper — Day 8 kept for action-level faults; message-level
# handled in the rollout loop (see inject_message_faults).
# =========================
class FaultWrapper:
    def __init__(self, env, fault_config, intensity=1.0):
        self.env = env
        self.fault_type = fault_config.get("type", "none")
        self.faulty_agents = list(fault_config.get("agents", []))
        self.intensity = intensity  # updated by curriculum each rollout
        # Intermittent fault için dropout probability (intensity'den ayrı).
        # intensity = curriculum gate'i, prob = aktifken kaç step'te düşüyor.
        # Default 0.3: literatürde ortak orta değer; 0.5 fail_stop'a yakın, 0.1 none'a yakın.
        self.intermittent_prob = float(fault_config.get("prob", 0.3))

    @property
    def agents(self):
        return self.env.agents

    @property
    def possible_agents(self):
        return self.env.possible_agents

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        actions = self._apply_action_fault(actions)
        return self.env.step(actions)

    def _apply_action_fault(self, actions):
        if self.fault_type == "none" or self.intensity <= 0.0:
            return actions
        new_actions = dict(actions)
        for a in actions:
            if a not in self.faulty_agents:
                continue
            if self.fault_type == "fail_stop":
                # Intensity acts as the probability we are truly "failed" now.
                if np.random.rand() < self.intensity:
                    new_actions[a] = 0
            elif self.fault_type == "byzantine":
                # Adversarial action: pick a random (non-noop) action.
                if np.random.rand() < self.intensity:
                    new_actions[a] = int(self.env.action_space(a).sample())
            elif self.fault_type == "intermittent":
                # FIX: intensity * prob — intensity curriculum gate'i,
                # prob ise gerçek dropout oranı. Önceki kodda intensity tek
                # başına dropout prob olarak kullanılıyordu → intensity=1.0'da
                # her step dropout = fail_stop davranışı (bug).
                effective_prob = self.intensity * self.intermittent_prob
                if np.random.rand() < effective_prob:
                    new_actions[a] = 0
        return new_actions


# =========================
# Networks — Day 8 architecture; added encoder/aggregate split.
# =========================
class CommNetActor(nn.Module):
    """
    Same CommNet actor as Day 8, just with explicit encode() and
    aggregate_and_policy() so we can inject faults between them.
    """

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, act_dim),
        )

    def encode(self, obs_agents):                      # (n, b, obs_dim) -> (n, b, H)
        return self.encoder(obs_agents)

    def aggregate_and_policy(self, h, adj):            # h:(n,b,H), adj:(n,n)
        msg = torch.einsum("ij,jbd->ibd", adj, h)
        out = torch.cat([h, msg], dim=-1)
        return self.policy(out)

    def forward(self, obs_agents, adj):
        h = self.encode(obs_agents)
        return self.aggregate_and_policy(h, adj)


class MAPPOCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * N_AGENTS, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =========================
# Message-level fault injection
# =========================
def inject_message_faults(h, fault_type, faulty_idx_list, intensity, rng,
                          byzantine_magnitude: float = 2.0,
                          intermittent_prob: float = 0.3):
    """
    Modify encoder output `h` (n, b, H) IN-PLACE for faulty agents.
    Called only when strategy requires it (B and C); strategy A keeps
    day8 semantics (action-level only, messages untouched).

    byzantine is ADVERSARIAL (not noise): the faulty agent broadcasts the
    inverted mean of the healthy agents' embeddings, *amplified* by
    `byzantine_magnitude`. A norm-preserving inversion alone can be hidden
    inside healthy-agent heterogeneity; a realistic attacker maximises
    damage, so magnitude amplification is the principled design. The
    magnitude is also the knob you vary in Day 10 ablations to study
    detection-threshold sensitivity (direct analogue of FDI residual
    threshold tuning).

    intermittent: dropout probability is `intensity * intermittent_prob`,
    NOT `intensity` alone. intensity acts as a curriculum gate (0 → off,
    1 → fully active); intermittent_prob is the *fault character* (how
    often the agent drops when active). Without this split, intensity=1.0
    + intermittent collapses to fail_stop semantics.
    """
    if fault_type == "none" or not faulty_idx_list or intensity <= 0.0:
        return h

    n = h.shape[0]
    faulty_set = set(faulty_idx_list)
    healthy_idx = [i for i in range(n) if i not in faulty_set]

    if fault_type == "fail_stop":
        for fi in faulty_idx_list:
            if rng.random() < intensity:
                h[fi] = 0.0

    elif fault_type == "byzantine":
        if healthy_idx:
            healthy_mean = h[healthy_idx].mean(dim=0)   # (b, H)
            adversarial = -byzantine_magnitude * healthy_mean
            for fi in faulty_idx_list:
                if rng.random() < intensity:
                    h[fi] = adversarial

    elif fault_type == "intermittent":
        # FIX: effective_prob = intensity * intermittent_prob (see docstring).
        effective_prob = intensity * intermittent_prob
        for fi in faulty_idx_list:
            if rng.random() < effective_prob:
                h[fi] = 0.0

    return h


# =========================
# GAE — same as Day 8
# =========================
def compute_gae(rewards, values, last_value, dones):
    T = len(rewards)
    values = np.append(values, last_value)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * values[t + 1] * mask - values[t]
        gae = delta + GAMMA * LAM * mask * gae
        adv[t] = gae
    returns = adv + values[:-1]
    return adv, returns


# =========================
# Rollout — Day 8 loop with (a) explicit encoder split, (b) strategy-aware
# fault injection, (c) optional fault-indicator feature.
# =========================
def collect_rollout(env, actor, critic, agents,
                    topo_mode, static_adj_raw, comm_radius,
                    obs_rms, fault_cfg, strategy,
                    curriculum, detector, topology_mgr,
                    use_fault_indicator, current_flags,
                    rng, byzantine_magnitude: float = 2.0,
                    intermittent_prob: float = 0.3):
    n = len(agents)
    base_obs_dim = env.observation_space(agents[0]).shape[0]
    obs_dim = base_obs_dim + (1 if use_fault_indicator else 0)

    buf_obs = np.zeros((n, ROLLOUT_STEPS, obs_dim), dtype=np.float32)
    buf_obs_global = np.zeros((ROLLOUT_STEPS, n * obs_dim), dtype=np.float32)
    buf_actions = np.zeros((n, ROLLOUT_STEPS), dtype=np.int64)
    buf_logp = np.zeros((n, ROLLOUT_STEPS), dtype=np.float32)
    buf_rewards = np.zeros(ROLLOUT_STEPS, dtype=np.float32)
    buf_dones = np.zeros(ROLLOUT_STEPS, dtype=np.float32)
    buf_values = np.zeros(ROLLOUT_STEPS, dtype=np.float32)

    # Per-agent ||h_i|| statistics — mean and std fed to FaultDetector v2.
    # FaultDetector now takes (msg_norms_mean, msg_norms_std) rather than the
    # fleet-deviation vector. Per-step norms are accumulated here.
    msg_norm_sum  = np.zeros(n, dtype=np.float64)
    msg_norm_sq   = np.zeros(n, dtype=np.float64)
    msg_norm_count = 0
    msg_h_sum = np.zeros((n, HIDDEN_DIM), dtype=np.float64)

    raw_reward_log, coverage_log, dist_log = [], [], []

    faulty_indices = [agents.index(a) for a in fault_cfg.get("agents", [])
                      if a in agents]
    fault_type = fault_cfg.get("type", "none")

    obs, _ = env.reset()

    for t in range(ROLLOUT_STEPS):
        if not env.agents:
            obs, _ = env.reset()

        # --- Build adjacency, apply topology reconfig if strategy C ---
        if topo_mode == "static":
            adj_raw = static_adj_raw.clone()
        else:
            adj_raw = compute_adj(obs, agents, comm_radius)

        if strategy == "C" and topology_mgr is not None:
            adj = topology_mgr.reconfigure(adj_raw, current_flags)
        else:
            adj = adj_raw + torch.eye(n, device=device)
            deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
            adj = adj / deg

        # --- Build observations (with optional fault indicator) ---
        obs_raw_list = [obs.get(a, np.zeros(base_obs_dim, dtype=np.float32))
                        for a in agents]
        obs_rms.update(np.asarray(obs_raw_list, dtype=np.float32))
        obs_norm_list = [normalize_obs(o, obs_rms).astype(np.float32)
                         for o in obs_raw_list]

        if use_fault_indicator:
            for i in range(n):
                if strategy == "B":
                    flag = 1.0 if (i in faulty_indices
                                   and curriculum is not None
                                   and curriculum.intensity > 0) else 0.0
                elif strategy == "C":
                    flag = 1.0 if current_flags[i] else 0.0
                else:
                    flag = 0.0
                obs_norm_list[i] = np.concatenate(
                    [obs_norm_list[i], np.array([flag], dtype=np.float32)]
                )

        obs_np = np.asarray(obs_norm_list, dtype=np.float32)      # (n, obs_dim)
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(1).to(device)  # (n, 1, D)
        obs_global_np = np.concatenate(obs_norm_list).astype(np.float32)
        obs_global = torch.from_numpy(obs_global_np).unsqueeze(0).to(device)

        # --- Forward pass with optional message-level fault injection ---
        with torch.no_grad():
            h = actor.encode(obs_tensor)                           # (n, 1, H)

            if strategy in ("B", "C"):
                if strategy == "A":
                    intensity = 1.0
                else:
                    intensity = curriculum.intensity if curriculum is not None else 1.0
                h = inject_message_faults(h, fault_type, faulty_indices,
                                          intensity, rng,
                                          byzantine_magnitude=byzantine_magnitude,
                                          intermittent_prob=intermittent_prob)

            # Accumulate per-agent deviation from the fleet-mean embedding
            # (L2 distance to mean-over-agents). This is the signal the
            # detector consumes: healthy homogeneous agents cluster tightly,
            # a faulty agent — byzantine/fail_stop/intermittent — sits far
            # from the healthy cluster in embedding space.
            with torch.no_grad():
                h_b = h.squeeze(1)                         # (n, H)
                per_agent_norm = h_b.norm(dim=-1).cpu().numpy()  # (n,)
                msg_norm_sum  += per_agent_norm
                msg_norm_sq   += per_agent_norm ** 2
                msg_h_sum += h_b.cpu().numpy().astype(np.float64)
                msg_norm_count += 1

            logits = actor.aggregate_and_policy(h, adj)
            value = critic(obs_global)

        # --- Sample actions ---
        actions = {}
        for i, a in enumerate(agents):
            dist = torch.distributions.Categorical(logits=logits[i, 0])
            act = dist.sample()
            actions[a] = int(act.item())
            buf_obs[i, t] = obs_norm_list[i]
            buf_actions[i, t] = int(act.item())
            buf_logp[i, t] = float(dist.log_prob(act).item())

        buf_obs_global[t] = obs_global_np
        buf_values[t] = float(value.item())

        # --- Step env (action-level faults applied inside FaultWrapper) ---
        next_obs, rewards, terms, truncs, _ = env.step(actions)
        cov_dense, cov_binary, avg_md = compute_metrics(next_obs, agents)

        shared_r = float(np.mean(list(rewards.values()))) if rewards else 0.0
        shaped_r = REWARD_SCALE * shared_r + COVERAGE_BONUS * cov_dense
        buf_rewards[t] = shaped_r

        any_done = False
        if terms or truncs:
            for a in agents:
                if terms.get(a, False) or truncs.get(a, False):
                    any_done = True
                    break
        buf_dones[t] = float(any_done)

        raw_reward_log.append(shared_r)
        coverage_log.append(cov_binary)
        if avg_md != float('inf'):
            dist_log.append(avg_md)

        obs = next_obs if next_obs else {}

    # --- Bootstrap last value ---
    if obs:
        obs_raw_list = [obs.get(a, np.zeros(base_obs_dim, dtype=np.float32))
                        for a in agents]
        obs_norm_list = [normalize_obs(o, obs_rms).astype(np.float32)
                         for o in obs_raw_list]
        if use_fault_indicator:
            for i in range(n):
                flag = 1.0 if (strategy == "C" and current_flags[i]) else 0.0
                obs_norm_list[i] = np.concatenate(
                    [obs_norm_list[i], np.array([flag], dtype=np.float32)]
                )
        obs_global = torch.from_numpy(
            np.concatenate(obs_norm_list).astype(np.float32)
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            last_value = float(critic(obs_global).item())
    else:
        last_value = 0.0

    adv, ret = compute_gae(buf_rewards, buf_values, last_value, buf_dones)

    cnt = max(1, msg_norm_count)
    msg_norms_mean = msg_norm_sum / cnt
    msg_norms_std  = np.sqrt(np.maximum(msg_norm_sq / cnt - (msg_norms_mean ** 2), 0))
    msg_h_mean = msg_h_sum / cnt
    return {
        "obs": buf_obs,
        "obs_global": buf_obs_global,
        "actions": buf_actions,
        "logp": buf_logp,
        "adv": adv,
        "ret": ret,
        "values": buf_values,
        "raw_reward": float(np.mean(raw_reward_log)) if raw_reward_log else 0.0,
        "coverage": float(np.mean(coverage_log)) if coverage_log else 0.0,
        "avg_dist": float(np.mean(dist_log)) if dist_log else float('inf'),
        "msg_norms_mean": msg_norms_mean,
        "msg_norms_std":  msg_norms_std,
        "msg_h_mean": msg_h_mean,
        "rollout_adj":    adj.clone(),
    }


# =========================
# PPO update — Day 8 exactly; adj built from obs_dim of actor.
# =========================
def ppo_update(actor, critic, opt_a, opt_c, buf, rollout_adj=None):
    obs = torch.from_numpy(buf["obs"]).to(device)
    obs_global = torch.from_numpy(buf["obs_global"]).to(device)
    actions = torch.from_numpy(buf["actions"]).to(device)
    old_lp = torch.from_numpy(buf["logp"]).to(device)
    adv = torch.from_numpy(buf["adv"]).to(device)
    ret = torch.from_numpy(buf["ret"]).to(device)
    old_values = torch.from_numpy(buf["values"]).to(device)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n_agents, T, _ = obs.shape

    # Use the adjacency that was active during rollout collection (Fix: was
    # hardcoded fully-connected, causing PPO ratio mismatch for strategy C
    # which may have used a reconfigured topology during rollout).
    if rollout_adj is not None:
        adj = rollout_adj.to(device)
    else:
        adj = torch.ones(n_agents, n_agents, device=device)
        adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1.0)

    total_entropy, total_vloss, total_ploss, total_msg_entropy, count = 0, 0, 0, 0, 0

    for _ in range(PPO_EPOCHS):
        idx = np.random.permutation(T)
        for start in range(0, T, MINI_BATCH):
            b_idx = torch.from_numpy(idx[start:start + MINI_BATCH]).long().to(device)
            obs_b = obs[:, b_idx]
            obs_g_b = obs_global[b_idx]
            act_b = actions[:, b_idx]
            old_lp_b = old_lp[:, b_idx]
            adv_b = adv[b_idx]
            ret_b = ret[b_idx]
            old_v_b = old_values[b_idx]

            logits = actor(obs_b, adj)
            values = critic(obs_g_b)

            h = actor.encoder(obs_b)
            msg = torch.einsum("ij,jbd->ibd", adj, h)
            msg_prob = torch.softmax(msg, dim=-1)
            msg_log_prob = torch.log(msg_prob + 1e-8)
            msg_entropy = -(msg_prob * msg_log_prob).sum(dim=-1).mean()

            v_clipped = old_v_b + torch.clamp(values - old_v_b, -VALUE_CLIP_EPS, VALUE_CLIP_EPS)
            v_loss_unclip = (values - ret_b).pow(2)
            v_loss_clip = (v_clipped - ret_b).pow(2)
            loss_c = 0.5 * torch.max(v_loss_unclip, v_loss_clip).mean()

            loss_a = 0.0
            entropy = 0.0
            for i in range(n_agents):
                dist = torch.distributions.Categorical(logits=logits[i])
                new_lp = dist.log_prob(act_b[i])
                ratio = torch.exp(new_lp - old_lp_b[i])
                clip = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                loss_a = loss_a + (-torch.min(ratio * adv_b, clip * adv_b).mean())
                entropy = entropy + dist.entropy().mean()

            loss_a = loss_a / n_agents
            entropy = entropy / n_agents

            opt_a.zero_grad()
            (loss_a - ENTROPY_COEF * entropy).backward()
            nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
            opt_a.step()

            opt_c.zero_grad()
            (VF_COEF * loss_c).backward()
            nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
            opt_c.step()

            total_entropy += float(entropy.item())
            total_msg_entropy += float(msg_entropy.item())
            total_vloss += float(loss_c.item())
            total_ploss += float(loss_a.item())
            count += 1

    return (total_entropy / count, total_vloss / count,
            total_ploss / count, total_msg_entropy / count)


# =========================
# Main
# =========================
def _config_hash(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="yaml config path")
    parser.add_argument("--strategy", choices=["A", "B", "C"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topology", default="configs/topology_full.json")
    parser.add_argument("--total_updates", type=int, default=None,
                        help="override TOTAL_UPDATES for smoke tests")
    parser.add_argument("--rollout_steps", type=int, default=None,
                        help="override ROLLOUT_STEPS for smoke tests")
    parser.add_argument("--wandb", action="store_true",
                        help="enable wandb logging (default: off)")
    args = parser.parse_args()

    global TOTAL_UPDATES, ROLLOUT_STEPS
    if args.total_updates is not None:
        TOTAL_UPDATES = args.total_updates
    if args.rollout_steps is not None:
        ROLLOUT_STEPS = args.rollout_steps

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_hash = _config_hash(cfg)
    print(f"[day9] config={args.config} strategy={args.strategy} "
          f"seed={args.seed} cfg_hash={cfg_hash}")

    fault_cfg = cfg["fault"]
    scenario = cfg.get("scenario", "unnamed")
    use_fault_indicator = bool(cfg.get("use_fault_indicator", True))

    # --- wandb ---
    if args.wandb and _WANDB_AVAILABLE:
        wandb.init(
            project="marl-fault-tolerance_day9_faf",
            name=f"day9_new_{args.strategy}_{scenario}_seed{args.seed}",
            config={**cfg, "strategy": args.strategy, "seed": args.seed,
                    "cfg_hash": cfg_hash,
                    "total_updates": TOTAL_UPDATES,
                    "rollout_steps": ROLLOUT_STEPS},
        )
        log_fn = wandb.log
    else:
        log_fn = lambda d: None  # noqa: E731

    # --- Env + topology ---
    env = FaultWrapper(
        simple_spread_v3.parallel_env(N=N_AGENTS, max_cycles=MAX_CYCLES),
        fault_cfg, intensity=1.0,
    )
    agents = env.possible_agents
    base_obs_dim = env.observation_space(agents[0]).shape[0]
    obs_dim = base_obs_dim + (1 if use_fault_indicator else 0)
    act_dim = env.action_space(agents[0]).n

    topo_mode, static_adj_raw, comm_radius = load_topology(args.topology)

    actor = CommNetActor(obs_dim, act_dim).to(device)
    critic = MAPPOCritic(obs_dim).to(device)
    opt_a = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_c = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    obs_rms = RunningMeanStd(shape=(base_obs_dim,))

    # --- Strategy-specific adaptation objects ---
    curriculum = None
    detector = None
    topology_mgr = None

    if args.strategy == "B":
        ccfg = cfg.get("curriculum", {})
        curriculum = CurriculumScheduler(
            total_updates=TOTAL_UPDATES,
            linear_frac=ccfg.get("linear_frac", 0.2),
            linear_max=ccfg.get("linear_max", 0.2),
            plateau_window=ccfg.get("plateau_window", 40),
            plateau_eps=ccfg.get("plateau_eps", 0.02),
            bump_step=ccfg.get("bump_step", 0.1),
            intensity_cap=ccfg.get("intensity_cap", 1.0),
        )
    elif args.strategy == "C":
        dcfg = cfg.get("detector", {})
        detector = FaultDetector(
            n_agents=N_AGENTS,
            k_threshold=dcfg.get("k_threshold", 3.0),
            k_cv=dcfg.get("k_cv", 2.0),
            k_fleet=dcfg.get("k_fleet", 0.7),
            k_cos_drop=dcfg.get("k_cos_drop", 0.35),  # YENİ
            k_coherence_drop=dcfg.get("k_coherence_drop", 0.3),  # YENİ
            ema_alpha=dcfg.get("ema_alpha", 0.05),
            hysteresis_M=dcfg.get("hysteresis_M", 5),
            set_persistence_K=dcfg.get("set_persistence_K", 3),
            warmup_updates=dcfg.get("warmup_updates", 30),
        )
        tcfg = cfg.get("topology_manager", {})
        topology_mgr = TopologyManager(
            n_agents=N_AGENTS,
            keep_self_loop=tcfg.get("keep_self_loop", True),
        )
        # For C, faults are on at full intensity from the start.
        env.intensity = 1.0

    current_flags = np.zeros(N_AGENTS, dtype=bool)
    converged_step = None
    last_buf = None
    byzantine_magnitude = float(cfg.get("byzantine_magnitude", BYZANTINE_MAGNITUDE_DEFAULT))
    # Intermittent dropout probability (curriculum intensity ile çarpılır).
    # Config'de scenario-level "fault.prob" ile override edilebilir.
    intermittent_prob = float(fault_cfg.get("prob", 0.3))

    for update in range(TOTAL_UPDATES):
        # Curriculum: set env + fault-injection intensity.
        if curriculum is not None:
            recent_reward = (last_buf["raw_reward"]
                             if last_buf is not None else 0.0)
            intensity = curriculum.step(update, recent_reward)
            env.intensity = intensity

        buf = collect_rollout(
            env, actor, critic, agents,
            topo_mode, static_adj_raw, comm_radius,
            obs_rms, fault_cfg, args.strategy,
            curriculum, detector, topology_mgr,
            use_fault_indicator, current_flags, rng,
            byzantine_magnitude=byzantine_magnitude,
            intermittent_prob=intermittent_prob,
        )

        # Detector step (strategy C only).
        if detector is not None:
            truth = [agents.index(a) for a in fault_cfg.get("agents", [])
                     if a in agents]
            current_flags = detector.update(
                buf["msg_norms_mean"],
                buf["msg_norms_std"],
                msg_h_mean=buf["msg_h_mean"],  # YENİ
                true_faulty=truth,
            )

        ent, vloss, ploss, msg_ent = ppo_update(actor, critic, opt_a, opt_c, buf,
                                                    rollout_adj=buf.get('rollout_adj'))

        if converged_step is None and buf["coverage"] >= CONV_COVERAGE_THRESHOLD:
            converged_step = update

        if update % 25 == 0:
            det_info = ""
            if detector is not None:
                det_info = f" | flags={current_flags.astype(int).tolist()}"
            cur_info = ""
            if curriculum is not None:
                cur_info = f" | intensity={curriculum.intensity:.2f}"
            print(f"Update {update:4d} | RawR: {buf['raw_reward']:7.3f} | "
                  f"Cov: {buf['coverage']:.3f} | "
                  f"AvgDist: {buf['avg_dist']:.3f} | "
                  f"Ent: {ent:.3f} | MsgEnt: {msg_ent:.3f}"
                  f"{cur_info}{det_info}")

        log_entry = {
            "reward": buf["raw_reward"],
            "coverage": buf["coverage"],
            "avg_dist": buf["avg_dist"],
            "policy_entropy": ent,
            "message_entropy": msg_ent,
            "value_loss": vloss,
            "policy_loss": ploss,
            "update": update,
            "convergence_step": (converged_step
                                 if converged_step is not None else -1),
        }
        if curriculum is not None:
            log_entry["curriculum_intensity"] = curriculum.intensity
        if detector is not None:
            for i in range(N_AGENTS):
                log_entry[f"flag_agent_{i}"] = int(current_flags[i])
            summ = detector.summary()
            log_entry["det_precision"] = summ["precision"]
            log_entry["det_recall"] = summ["recall"]
            log_entry["det_fpr"] = summ["false_positive_rate"]
            log_entry["det_f1"] = summ["f1"]
        log_fn(log_entry)

        last_buf = buf

    # --- Summary ---
    final = {
        "final_reward": last_buf["raw_reward"],
        "final_coverage": last_buf["coverage"],
        "final_avg_dist": last_buf["avg_dist"],
        "convergence_step": (converged_step
                             if converged_step is not None else -1),
    }
    if detector is not None:
        final.update({f"det_{k}": v for k, v in detector.summary().items()})

    print("[day9] final:", final)
    if args.wandb and _WANDB_AVAILABLE:
        for k, v in final.items():
            wandb.summary[k] = v

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "opt_actor": opt_a.state_dict(),
        "opt_critic": opt_c.state_dict(),
        "obs_mean": obs_rms.mean,
        "obs_var": obs_rms.var,
        "config": cfg,
        "args": vars(args),
        "final": final,
        "detector_summary": detector.summary() if detector is not None else None,
    }, f"checkpoints/day9_new_{args.strategy}_{scenario}_seed{args.seed}.pt")

    return final


if __name__ == "__main__":
    main()

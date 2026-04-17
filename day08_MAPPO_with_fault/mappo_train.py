import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from mpe2 import simple_spread_v3
import wandb

# =========================
# HYPERPARAMETERS
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
PPO_EPOCHS = 10
MINI_BATCH = 256

ROLLOUT_STEPS = 1000
TOTAL_UPDATES = 3000

ENTROPY_COEF = 0.005
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

COVERAGE_THRESHOLD = 0.30
COVERAGE_BONUS = 1.0        # shared global shaping
REWARD_SCALE = 1.0          # raw reward ölçeği

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x)
        if x.ndim == 1 and self.mean.ndim == 0:
            batch_mean = x.mean()
            batch_var = x.var()
            batch_count = x.size
        else:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        new_var = M2 / tot
        self.mean = new_mean
        self.var = new_var
        self.count = tot


class FaultWrapper:
    def __init__(self, env, fault_config):
        self.env = env
        self.fault_type = fault_config.get("type", "none")
        self.faulty_agents = fault_config.get("agents", [])

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
        if self.fault_type == "none":
            return actions
        new_actions = dict(actions)
        for a in actions:
            if a not in self.faulty_agents:
                continue
            if self.fault_type == "fail_stop":
                new_actions[a] = 0
            elif self.fault_type == "byzantine":
                new_actions[a] = self.env.action_space(a).sample()
            elif self.fault_type == "intermittent":
                if np.random.rand() < 0.5:
                    new_actions[a] = 0
        return new_actions


def load_topology(path):
    with open(path, "r") as f:
        topo = json.load(f)
    if topo["mode"] == "static":
        adj = torch.tensor(topo["adj_matrix"], dtype=torch.float32).to(device)
        return "static", adj, None
    else:
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
            rel = o[other_start + idx*2: other_start + idx*2 + 2]
            if np.linalg.norm(rel) < r:
                adj[i, j] = 1.0
            idx += 1
    return adj


def compute_metrics(obs, agents, threshold=COVERAGE_THRESHOLD):
    if not obs:
        return 0.0, float('inf')
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
        return 0.0, float('inf')
    coverage = sum(1 for d in valid if d < threshold) / N_AGENTS
    avg_min_dist = float(np.mean(valid))
    return coverage, avg_min_dist


class CommNetActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, act_dim)
        )

    def forward(self, obs_agents, adj):
        n, b, _ = obs_agents.shape
        h = self.encoder(obs_agents)
        msg = torch.einsum("ij,jbd->ibd", adj, h)
        out = torch.cat([h, msg], dim=-1)
        logits = self.policy(out)
        return logits


class MAPPOCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * N_AGENTS, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_gae(rewards, values, last_value, dones):
    T = len(rewards)
    values = np.append(values, last_value)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * values[t+1] * mask - values[t]
        gae = delta + GAMMA * LAM * mask * gae
        adv[t] = gae
    returns = adv + values[:-1]
    return adv, returns


def normalize_obs(obs_array, obs_rms, eps=1e-8):
    return (obs_array - obs_rms.mean) / (np.sqrt(obs_rms.var) + eps)


def collect_rollout(env, actor, critic, agents,
                    topo_mode, static_adj, comm_radius,
                    obs_rms):

    n = len(agents)
    obs_dim = env.observation_space(agents[0]).shape[0]

    buf_obs = np.zeros((n, ROLLOUT_STEPS, obs_dim), dtype=np.float32)
    buf_obs_global = np.zeros((ROLLOUT_STEPS, n * obs_dim), dtype=np.float32)
    buf_actions = np.zeros((n, ROLLOUT_STEPS), dtype=np.int64)
    buf_logp = np.zeros((n, ROLLOUT_STEPS), dtype=np.float32)
    buf_rewards = np.zeros(ROLLOUT_STEPS, dtype=np.float32)   # SHARED
    buf_dones = np.zeros(ROLLOUT_STEPS, dtype=np.float32)
    buf_values = np.zeros(ROLLOUT_STEPS, dtype=np.float32)

    raw_reward_log = []
    coverage_log = []
    dist_log = []

    obs, _ = env.reset()

    for t in range(ROLLOUT_STEPS):
        if not env.agents:
            obs, _ = env.reset()

        if topo_mode == "static":
            adj = static_adj.clone()
        else:
            adj = compute_adj(obs, agents, comm_radius)

        adj = adj + torch.eye(n, device=device)
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        adj = adj / deg

        obs_list_raw = [obs.get(a, np.zeros(obs_dim, dtype=np.float32)) for a in agents]

        obs_rms.update(np.array(obs_list_raw, dtype=np.float32))
        obs_list_norm = [normalize_obs(o, obs_rms).astype(np.float32) for o in obs_list_raw]

        obs_np_norm = np.array(obs_list_norm, dtype=np.float32)
        obs_tensor = torch.from_numpy(obs_np_norm).unsqueeze(1).to(device)
        obs_global_np = np.concatenate(obs_list_norm).astype(np.float32)
        obs_global = torch.from_numpy(obs_global_np).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = actor(obs_tensor, adj)
            value = critic(obs_global)

        actions = {}
        for i, a in enumerate(agents):
            dist = torch.distributions.Categorical(logits=logits[i, 0])
            act = dist.sample()
            actions[a] = act.item()
            buf_obs[i, t] = obs_list_norm[i]
            buf_actions[i, t] = act.item()
            buf_logp[i, t] = dist.log_prob(act).item()

        buf_obs_global[t] = obs_global_np
        buf_values[t] = value.item()

        next_obs, rewards, terms, truncs, _ = env.step(actions)

        cov, avg_md = compute_metrics(next_obs, agents)

        # COOPERATIVE / SHARED reward: simple_spread'de tüm ajanlar aynı reward alır
        if rewards:
            shared_r = float(np.mean(list(rewards.values())))
        else:
            shared_r = 0.0

        # Shaped reward = scaled_raw + shared coverage bonus
        # Her iki terim de TÜM ajanlara aynı gider → cooperative
        shaped_r = REWARD_SCALE * shared_r + COVERAGE_BONUS * cov

        buf_rewards[t] = shaped_r

        any_done = False
        if terms or truncs:
            for a in agents:
                if terms.get(a, False) or truncs.get(a, False):
                    any_done = True
                    break
        buf_dones[t] = float(any_done)

        raw_reward_log.append(shared_r)
        coverage_log.append(cov)
        if avg_md != float('inf'):
            dist_log.append(avg_md)

        obs = next_obs if next_obs else {}

    if obs:
        obs_list_raw = [obs.get(a, np.zeros(obs_dim, dtype=np.float32)) for a in agents]
        obs_list_norm = [normalize_obs(o, obs_rms).astype(np.float32) for o in obs_list_raw]
        obs_global = torch.from_numpy(np.concatenate(obs_list_norm).astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            last_value = critic(obs_global).item()
    else:
        last_value = 0.0

    # Tek GAE hesabı — shared reward, shared advantage
    adv, ret = compute_gae(buf_rewards, buf_values, last_value, buf_dones)

    return {
        "obs": buf_obs,
        "obs_global": buf_obs_global,
        "actions": buf_actions,
        "logp": buf_logp,
        "adv": adv,               # (T,) shared
        "ret": ret,               # (T,)
        "values": buf_values,
        "raw_reward": float(np.mean(raw_reward_log)) if raw_reward_log else 0.0,
        "coverage": float(np.mean(coverage_log)) if coverage_log else 0.0,
        "avg_dist": float(np.mean(dist_log)) if dist_log else float('inf'),
    }


def ppo_update(actor, critic, opt_a, opt_c, buf):
    obs = torch.from_numpy(buf["obs"]).to(device)
    obs_global = torch.from_numpy(buf["obs_global"]).to(device)
    actions = torch.from_numpy(buf["actions"]).to(device)
    old_lp = torch.from_numpy(buf["logp"]).to(device)
    adv = torch.from_numpy(buf["adv"]).to(device)       # (T,) shared
    ret = torch.from_numpy(buf["ret"]).to(device)       # (T,)
    old_values = torch.from_numpy(buf["values"]).to(device)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n_agents, T, _ = obs.shape

    adj = torch.ones(n_agents, n_agents, device=device)
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    adj = adj / deg

    total_entropy = 0.0
    total_vloss = 0.0
    total_ploss = 0.0
    total_msg_entropy = 0.0
    count = 0

    for _ in range(PPO_EPOCHS):
        idx = np.random.permutation(T)
        for start in range(0, T, MINI_BATCH):
            b_idx = torch.from_numpy(idx[start:start+MINI_BATCH]).long().to(device)
            msg_entropy = 0.0
            obs_b = obs[:, b_idx]
            obs_g_b = obs_global[b_idx]
            act_b = actions[:, b_idx]
            old_lp_b = old_lp[:, b_idx]
            adv_b = adv[b_idx]              # (B,) — herkes aynı
            ret_b = ret[b_idx]
            old_v_b = old_values[b_idx]

            logits = actor(obs_b, adj)
            values = critic(obs_g_b)

            # message entropy (hidden state üzerinden approx)
            h = actor.encoder(obs_b)  # (n_agents, B, hidden)
            msg = torch.einsum("ij,jbd->ibd", adj, h)

            msg_prob = torch.softmax(msg, dim=-1)
            msg_log_prob = torch.log(msg_prob + 1e-8)

            msg_entropy += -(msg_prob * msg_log_prob).sum(dim=-1).mean()

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
                # Tüm ajanlar aynı shared advantage'ı kullanır → cooperative
                loss_a = loss_a + (-torch.min(ratio * adv_b, clip * adv_b).mean())
                entropy = entropy + dist.entropy().mean()

            loss_a = loss_a / n_agents
            entropy = entropy / n_agents
            msg_entropy = msg_entropy / n_agents
            opt_a.zero_grad()
            (loss_a - ENTROPY_COEF * entropy).backward()
            nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
            opt_a.step()

            opt_c.zero_grad()
            (VF_COEF * loss_c).backward()
            nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
            opt_c.step()

            total_entropy += entropy.item()
            total_msg_entropy += msg_entropy.item()
            total_vloss += loss_c.item()
            total_ploss += loss_a.item()
            count += 1

    return total_entropy / count, total_vloss / count, total_ploss / count, total_msg_entropy / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault_type", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topology", default="topology_full.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    wandb.init(
        project="marl-fault-tolerance",
        name=f"{args.fault_type}_seed{args.seed}",
        config={
            "fault_type": args.fault_type,
            "seed": args.seed,
            "lr_actor": LR_ACTOR,
            "lr_critic": LR_CRITIC,
            "gamma": GAMMA,
            "clip_eps": CLIP_EPS,
        }
    )
    with open(f"configs/{args.fault_type}.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    fault_config = cfg["faults"][args.fault_type]
    topo_mode, static_adj, comm_radius = load_topology(args.topology)

    env = FaultWrapper(
        simple_spread_v3.parallel_env(N=N_AGENTS, max_cycles=MAX_CYCLES),
        fault_config
    )

    agents = env.possible_agents
    obs_dim = env.observation_space(agents[0]).shape[0]
    act_dim = env.action_space(agents[0]).n

    actor = CommNetActor(obs_dim, act_dim).to(device)
    critic = MAPPOCritic(obs_dim).to(device)

    opt_a = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_c = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    converged_step = None
    CONV_THRESHOLD = 0.6
    for update in range(TOTAL_UPDATES):
        buf = collect_rollout(env, actor, critic, agents,
                              topo_mode, static_adj, comm_radius,
                              obs_rms)
        if converged_step is None and buf["coverage"] >= CONV_THRESHOLD:
            converged_step = update
        ent, vloss, ploss, msg_ent = ppo_update(actor, critic, opt_a, opt_c, buf)

        if update % 25 == 0:
            print(f"Update {update:4d} | RawR: {buf['raw_reward']:7.3f} | "
                  f"Cov: {buf['coverage']:.3f} | "
                  f"AvgDist: {buf['avg_dist']:.3f} | "
                  f"Ent: {ent:.3f} | "
                  f"VL: {vloss:.4f} | PL: {ploss:.4f} |"
                  f"MsgEnt: {msg_ent:.3f}")

        wandb.log({
            "reward": buf["raw_reward"],
            "coverage": buf["coverage"],
            "avg_dist": buf["avg_dist"],
            "policy_entropy": ent,
            "message_entropy": msg_ent,
            "value_loss": vloss,
            "policy_loss": ploss,
            "update": update,
            "converged_now": 1 if buf["coverage"] >= CONV_THRESHOLD else 0,
            "convergence_step": converged_step if converged_step is not None else -1,
        })

    wandb.summary["final_reward"] = buf["raw_reward"]
    wandb.summary["final_coverage"] = buf["coverage"]
    wandb.summary["final_avg_dist"] = buf["avg_dist"]
    wandb.summary["final_policy_entropy"] = ent
    wandb.summary["final_value_loss"] = vloss
    wandb.summary["convergence_step"] = converged_step if converged_step is not None else -1
    os.makedirs("checkpoints", exist_ok=True)
    #torch.save(actor.state_dict(), f"checkpoints/model_{args.fault_type}_seed{args.seed}.pt")
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "opt_actor": opt_a.state_dict(),
        "opt_critic": opt_c.state_dict(),
        "obs_mean": obs_rms.mean,
        "obs_var": obs_rms.var,
        "config": vars(args),
    }, f"checkpoints/model_{args.fault_type}_seed{args.seed}.pt")


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from mpe2 import simple_spread_v3

# =========================
# 1. HYPERPARAMETERS
# =========================
N_AGENTS      = 3
MAX_CYCLES    = 32
HIDDEN_DIM    = 128
LR            = 3e-4
GAMMA         = 0.99
LAM           = 0.95
CLIP_EPS      = 0.2
PPO_EPOCHS    = 10
MINI_BATCH    = 256
ROLLOUT_STEPS = 512
TOTAL_UPDATES = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. TOPOLOGY LOADER
# =========================
def load_topology(path):
    with open(path, "r") as f:
        topo = json.load(f)

    mode = topo["mode"]

    if mode == "static":
        adj = torch.tensor(topo["adj_matrix"], dtype=torch.float32).to(device)
        print(f"Topology : static — {topo['description']}")
        print(f"ADJ matrix:\n{adj.cpu().numpy()}\n")
        return mode, adj, None

    elif mode == "dynamic":
        r = topo["comm_radius"]
        print(f"Topology : dynamic — {topo['description']}")
        print(f"Comm radius: {r}\n")
        return mode, None, r

    else:
        raise ValueError(f"Bilinmeyen topology mode: {mode}")


# =========================
# 3. MODELS
# =========================
class CommNetActor(nn.Module):
    def __init__(self, obs_dim, act_dim, n_agents, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.n_agents   = n_agents
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh()
        )

        self.actor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim)
        )

    def message_passing(self, obs_agents, adj):
        # obs_agents: [n_agents, batch_size, obs_dim]
        h = torch.stack([
            self.encoder(obs_agents[i]) for i in range(self.n_agents)
        ])
        # h: [n_agents, batch_size, hidden_dim]

        messages = []
        for i in range(self.n_agents):
            neighbor_idx = adj[i].nonzero(as_tuple=True)[0]
            if len(neighbor_idx) == 0:
                m_i = torch.zeros_like(h[i])
            else:
                neighbor_h = torch.stack([h[j] for j in neighbor_idx])
                m_i = neighbor_h.mean(dim=0)
            messages.append(m_i)

        msg = torch.stack(messages)
        return h, msg

    def forward(self, obs_agents, adj):
        h, msg   = self.message_passing(obs_agents, adj)
        combined = torch.cat([h, msg], dim=-1)
        logits   = torch.stack([
            self.actor(combined[i]) for i in range(self.n_agents)
        ])
        # logits: [n_agents, batch_size, act_dim]
        return logits


class MAPPOCritic(nn.Module):
    def __init__(self, obs_dim, n_agents, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(n_agents * obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_agents)
        )

    def forward(self, obs_all):
        # obs_all: [batch_size, n_agents*obs_dim]
        # output:  [batch_size, n_agents]
        return self.critic(obs_all)


# =========================
# 4. COMPUTE ADJ (dynamic)
# =========================
def compute_adj(obs, agents, r):
    n   = len(agents)
    adj = torch.zeros(n, n)

    for i in range(n):
        other_indices = [k for k in range(n) if k != i]
        for j, other_idx in enumerate(other_indices):
            rel_pos = obs[agents[i]][10 + j*2 : 10 + j*2 + 2]
            dist    = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
            if dist < r:
                adj[i][other_idx] = 1.0

    return adj.to(device)


# =========================
# 5. GAE
# =========================
def compute_gae(rewards, values, last_value, dones, gamma=GAMMA, lam=LAM):
    values_ext = np.append(values, last_value)
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv   = 0.0

    for t in reversed(range(T)):
        delta         = rewards[t] + gamma * values_ext[t+1] * (1 - dones[t]) - values_ext[t]
        last_adv      = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns


# =========================
# 6. ROLLOUT
# =========================
def collect_rollout(env, actor, critic, agents, device,
                    topo_mode, static_adj, comm_radius,
                    rollout_steps=ROLLOUT_STEPS):

    buffer = {a: {"obs": [], "actions": [], "rewards": [],
                  "dones": [], "log_probs": [], "values": []}
              for a in agents}

    obs, _ = env.reset()

    for _ in range(rollout_steps):
        if not env.agents:
            obs, _ = env.reset()

        # Topology mode'a göre ADJ
        if topo_mode == "static":
            adj = static_adj
        else:
            adj = compute_adj(obs, agents, comm_radius)

        # Actor input — [n_agents, 1, obs_dim]
        obs_tensor = torch.stack([
            torch.tensor(obs[a], dtype=torch.float32).to(device)
            for a in agents
        ]).unsqueeze(1)

        # Critic input — [1, n_agents*obs_dim]
        obs_concat = torch.cat([
            torch.tensor(obs[a], dtype=torch.float32).to(device)
            for a in agents
        ]).unsqueeze(0)

        with torch.no_grad():
            logits = actor(obs_tensor, adj)   # [n_agents, 1, act_dim]
            values = critic(obs_concat)        # [1, n_agents]

        actions, log_probs = {}, {}
        for i, agent in enumerate(agents):
            dist             = torch.distributions.Categorical(logits=logits[i, 0])
            a                = dist.sample()
            actions[agent]   = a.item()
            log_probs[agent] = dist.log_prob(a).item()

        next_obs, rewards, terms, truncs, _ = env.step(actions)

        for i, agent in enumerate(agents):
            done = terms.get(agent, False) or truncs.get(agent, False)
            buffer[agent]["obs"].append(obs[agent])
            buffer[agent]["actions"].append(actions[agent])
            buffer[agent]["rewards"].append(rewards.get(agent, 0.0))
            buffer[agent]["dones"].append(float(done))
            buffer[agent]["log_probs"].append(log_probs[agent])
            buffer[agent]["values"].append(values[0, i].item())

        obs = next_obs if next_obs else {}

    # GAE
    all_obs, all_actions, all_lps = [], [], []
    all_adv, all_ret              = [], []

    for i, agent in enumerate(agents):
        r = np.array(buffer[agent]["rewards"], dtype=np.float32)
        v = np.array(buffer[agent]["values"],  dtype=np.float32)
        d = np.array(buffer[agent]["dones"],   dtype=np.float32)

        if obs and agent in obs:
            if topo_mode == "static":
                last_adj = static_adj
            else:
                last_adj = compute_adj(obs, agents, comm_radius)

            last_obs_concat = torch.cat([
                torch.tensor(obs[a], dtype=torch.float32).to(device)
                for a in agents
            ]).unsqueeze(0)

            with torch.no_grad():
                last_values = critic(last_obs_concat)
                last_value  = last_values[0, i].item()
        else:
            last_value = 0.0

        adv, ret = compute_gae(r, v, last_value, d)
        adv      = (adv - adv.mean()) / (adv.std() + 1e-8)

        all_obs.append(np.array(buffer[agent]["obs"]))
        all_actions.append(np.array(buffer[agent]["actions"]))
        all_lps.append(np.array(buffer[agent]["log_probs"]))
        all_adv.append(adv)
        all_ret.append(ret)

    combined = {
        "obs":               np.concatenate(all_obs),
        "actions":           np.concatenate(all_actions),
        "log_probs":         np.concatenate(all_lps),
        "advantages":        np.concatenate(all_adv),
        "returns":           np.concatenate(all_ret),
        "per_agent_rewards": {a: buffer[a]["rewards"] for a in agents}
    }

    return combined


# =========================
# 7. PPO UPDATE
# =========================
def ppo_update(actor, critic, actor_opt, critic_opt, buf, agents, device,
               ppo_epochs=PPO_EPOCHS, mini_batch_size=MINI_BATCH,
               clip_eps=CLIP_EPS, entropy_coef=0.05):

    obs        = torch.tensor(buf["obs"],        dtype=torch.float32).to(device)
    actions    = torch.tensor(buf["actions"],    dtype=torch.long).to(device)
    old_lps    = torch.tensor(buf["log_probs"],  dtype=torch.float32).to(device)
    advantages = torch.tensor(buf["advantages"], dtype=torch.float32).to(device)
    returns    = torch.tensor(buf["returns"],    dtype=torch.float32).to(device)

    n_agents  = len(agents)
    T         = ROLLOUT_STEPS

    # Training'de tam bağlı ADJ — centralized training
    train_adj = (torch.ones(n_agents, n_agents) - torch.eye(n_agents)).to(device)

    for _ in range(ppo_epochs):
        perm = np.random.permutation(T)

        for start in range(0, T, mini_batch_size):
            idx   = perm[start:start + mini_batch_size]
            idx_t = torch.tensor(idx, dtype=torch.long).to(device)

            # Actor input — [n_agents, mini_batch, obs_dim]
            obs_agents = torch.stack([
                obs[i * T:(i + 1) * T][idx_t]
                for i in range(n_agents)
            ])

            # Critic input — [mini_batch, n_agents*obs_dim]
            obs_concat = torch.cat([
                obs[i * T:(i + 1) * T][idx_t]
                for i in range(n_agents)
            ], dim=-1)

            logits      = actor(obs_agents, train_adj)
            logits_flat = logits.reshape(-1, logits.shape[-1])

            values_all  = critic(obs_concat)
            values_flat = values_all.reshape(-1)

            all_idx   = torch.cat([idx_t + i * T for i in range(n_agents)])

            dist        = torch.distributions.Categorical(logits=logits_flat)
            new_lps     = dist.log_prob(actions[all_idx])
            ratio       = torch.exp(new_lps - old_lps[all_idx])
            clipped     = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            adv_batch   = advantages[all_idx]

            actor_loss  = -torch.min(ratio * adv_batch, clipped * adv_batch).mean()
            entropy     = dist.entropy().mean()
            critic_loss = nn.functional.mse_loss(values_flat, returns[all_idx])

            actor_opt.zero_grad()
            (actor_loss - entropy_coef * entropy).backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_opt.step()

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_opt.step()


# =========================
# 8. PLOT
# =========================
def plot_rewards(reward_history, n_agents, topo_name=""):
    plt.figure(figsize=(12, 5))
    plt.plot(reward_history, alpha=0.3, label="Raw")

    window   = 20
    smoothed = np.convolve(
        reward_history, np.ones(window) / window, mode='valid'
    )
    plt.plot(range(window - 1, len(reward_history)), smoothed,
             linewidth=2, label=f"Smoothed ({window})")

    plt.xlabel("Update")
    plt.ylabel("Mean Episode Reward")
    plt.title(f"MAPPO — simple_spread (N={n_agents}) — {topo_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"reward_curve_mappo_{topo_name}.png", dpi=150)
    plt.show()

    print(f"\nİlk 20 update ort : {np.mean(reward_history[:20]):.3f}")
    print(f"Son  20 update ort : {np.mean(reward_history[-20:]):.3f}")
    print(f"En iyi update      : {max(reward_history):.3f}")


# =========================
# 9. MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="topology_full.json",
                        help="Topology config dosyası")
    args = parser.parse_args()

    topo_mode, static_adj, comm_radius = load_topology(args.topology)

    env    = simple_spread_v3.parallel_env(N=N_AGENTS, max_cycles=MAX_CYCLES)
    obs, _ = env.reset()
    agents = env.possible_agents

    obs_dim = env.observation_space(agents[0]).shape[0]
    act_dim = env.action_space(agents[0]).n

    print(f"Device  : {device}")
    print(f"Agents  : {agents}")
    print(f"Obs dim : {obs_dim}  |  Act dim: {act_dim}")
    print(f"Toplam timestep: {TOTAL_UPDATES * ROLLOUT_STEPS * N_AGENTS:,}\n")

    actor      = CommNetActor(obs_dim, act_dim, N_AGENTS).to(device)
    critic     = MAPPOCritic(obs_dim, N_AGENTS).to(device)
    actor_opt  = optim.Adam(actor.parameters(),  lr=LR)
    critic_opt = optim.Adam(critic.parameters(), lr=LR)

    reward_history       = []
    episodes_per_rollout = ROLLOUT_STEPS // MAX_CYCLES

    for update in range(TOTAL_UPDATES):
        buf = collect_rollout(env, actor, critic, agents, device,
                              topo_mode, static_adj, comm_radius)
        ppo_update(actor, critic, actor_opt, critic_opt, buf, agents, device)

        mean_reward = np.mean([
            sum(buf["per_agent_rewards"][a]) for a in agents
        ]) / episodes_per_rollout
        reward_history.append(mean_reward)

        if update % 100 == 0:
            print(f"Update {update:4d}/{TOTAL_UPDATES} | Mean Reward: {mean_reward:.3f}")

    topo_name = args.topology.replace(".json", "").replace("topology_", "")
    torch.save(actor.state_dict(),  f"mappo_actor_{topo_name}.pth")
    torch.save(critic.state_dict(), f"mappo_critic_{topo_name}.pth")
    print(f"Model kaydedildi: mappo_actor_{topo_name}.pth, mappo_critic_{topo_name}.pth")

    plot_rewards(reward_history, N_AGENTS, topo_name)


if __name__ == "__main__":
    main()
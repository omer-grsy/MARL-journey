import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpe2 import simple_spread_v3

# =========================
# 1. HYPERPARAMETERS
# =========================
N_AGENTS      = 3
MAX_CYCLES    = 32
HIDDEN_DIM    = 128
LR            = 1e-4
GAMMA         = 0.99
LAM           = 0.95
CLIP_EPS      = 0.2
PPO_EPOCHS    = 5
MINI_BATCH    = 256
ROLLOUT_STEPS = 512
TOTAL_UPDATES = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ADJ = torch.tensor([
#     [0, 1, 1],
#     [1, 0, 1],
#     [1, 1, 0]
# ], dtype=torch.float32).to(device)
COMM_RADIUS = 1.0


# =========================
# 2. MODEL
# =========================
class CommNetActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, n_agents, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.n_agents  = n_agents
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

        self.critic = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
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
        # msg: [n_agents, batch_size, hidden_dim]
        return h, msg

    def forward(self, obs_agents, adj):
        # obs_agents: [n_agents, batch_size, obs_dim]
        h, msg = self.message_passing(obs_agents, adj)

        combined = torch.cat([h, msg], dim=-1)
        # combined: [n_agents, batch_size, 2*hidden_dim]

        logits = torch.stack([
            self.actor(combined[i]) for i in range(self.n_agents)
        ])
        values = torch.stack([
            self.critic(combined[i]) for i in range(self.n_agents)
        ])
        # logits: [n_agents, batch_size, act_dim]
        # values: [n_agents, batch_size, 1]

        return logits, values.squeeze(-1)


# ===============================
# ADJ hesaplama - Distance based
# ===============================
def compute_adj(obs, agents, r=COMM_RADIUS):
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
# 3. GAE
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
# 4. ROLLOUT
# =========================
def collect_rollout(env, policy, agents, device, rollout_steps=ROLLOUT_STEPS):
    buffer = {a: {"obs": [], "actions": [], "rewards": [],
                  "dones": [], "log_probs": [], "values": []}
              for a in agents}

    obs, _ = env.reset()

    for _ in range(rollout_steps):
        if not env.agents:
            obs, _ = env.reset()

        adj = compute_adj(obs, agents) # her döngüde komşuluk hesapla
        # [n_agents, 1, obs_dim]
        obs_tensor = torch.stack([
            torch.tensor(obs[a], dtype=torch.float32).to(device)
            for a in agents
        ]).unsqueeze(1)

        with torch.no_grad():
            logits, values = policy(obs_tensor, adj)
            # logits: [n_agents, 1, act_dim]
            # values: [n_agents, 1]

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
            buffer[agent]["values"].append(values[i, 0].item())

        obs = next_obs if next_obs else {}

    # GAE
    all_obs, all_actions, all_lps = [], [], []
    all_adv, all_ret              = [], []

    for i, agent in enumerate(agents):
        r = np.array(buffer[agent]["rewards"],  dtype=np.float32)
        v = np.array(buffer[agent]["values"],   dtype=np.float32)
        d = np.array(buffer[agent]["dones"],    dtype=np.float32)

        if obs and agent in obs:
            last_adj = compute_adj(obs,agents)

            last_obs_all = torch.stack([
                torch.tensor(obs[a], dtype=torch.float32).to(device)
                for a in agents
            ]).unsqueeze(1)

            with torch.no_grad():
                _, last_values = policy(last_obs_all, last_adj)
                last_value     = last_values[i, 0].item()
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
        "obs":               np.concatenate(all_obs),      # [n_agents*512, obs_dim]
        "actions":           np.concatenate(all_actions),  # [n_agents*512]
        "log_probs":         np.concatenate(all_lps),      # [n_agents*512]
        "advantages":        np.concatenate(all_adv),      # [n_agents*512]
        "returns":           np.concatenate(all_ret),      # [n_agents*512]
        "per_agent_rewards": {a: buffer[a]["rewards"] for a in agents}
    }

    return combined


# =========================
# 5. PPO UPDATE
# =========================
def ppo_update(policy, optimizer, buf, agents, device,
               ppo_epochs=PPO_EPOCHS, mini_batch_size=MINI_BATCH,
               clip_eps=CLIP_EPS, value_coef=0.5, entropy_coef=0.05):

    # Training'de tam bağlı ADJ — centralized training
    n_agents  = len(agents)
    train_adj = (torch.ones(n_agents, n_agents) - torch.eye(n_agents)).to(device)
    # [[0,1,1],
    #  [1,0,1],
    #  [1,1,0]]  — kendine mesaj yok, herkese bağlı

    obs        = torch.tensor(buf["obs"],        dtype=torch.float32).to(device)
    actions    = torch.tensor(buf["actions"],    dtype=torch.long).to(device)
    old_lps    = torch.tensor(buf["log_probs"],  dtype=torch.float32).to(device)
    advantages = torch.tensor(buf["advantages"], dtype=torch.float32).to(device)
    returns    = torch.tensor(buf["returns"],    dtype=torch.float32).to(device)

    n_agents = len(agents)
    T        = ROLLOUT_STEPS

    for _ in range(ppo_epochs):
        perm = np.random.permutation(T)

        for start in range(0, T, mini_batch_size):
            idx   = perm[start:start + mini_batch_size]
            idx_t = torch.tensor(idx, dtype=torch.long).to(device)

            obs_agents = torch.stack([
                obs[i * T:(i + 1) * T][idx_t]
                for i in range(n_agents)
            ])
            # obs_agents: [n_agents, mini_batch, obs_dim]

            logits, values = policy(obs_agents, train_adj)

            logits_flat = logits.reshape(-1, logits.shape[-1])
            values_flat = values.reshape(-1)

            all_idx = torch.cat([
                idx_t + i * T for i in range(n_agents)
            ])

            dist        = torch.distributions.Categorical(logits=logits_flat)
            new_lps     = dist.log_prob(actions[all_idx])
            ratio       = torch.exp(new_lps - old_lps[all_idx])
            clipped     = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            adv_batch   = advantages[all_idx]

            actor_loss  = -torch.min(ratio * adv_batch, clipped * adv_batch).mean()
            critic_loss = nn.functional.mse_loss(values_flat, returns[all_idx])
            entropy     = dist.entropy().mean()

            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


# =========================
# 6. PLOT
# =========================
def plot_rewards(reward_history, n_agents):
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
    plt.title(f"CommNet — simple_spread (N={n_agents})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reward_curve_commnet-dyn.png", dpi=150)
    plt.show()

    print(f"\nİlk 20 update ort : {np.mean(reward_history[:20]):.3f}")
    print(f"Son  20 update ort : {np.mean(reward_history[-20:]):.3f}")
    print(f"En iyi update      : {max(reward_history):.3f}")


# =========================
# 7. MAIN
# =========================
def main():
    env    = simple_spread_v3.parallel_env(N=N_AGENTS, max_cycles=MAX_CYCLES)
    obs, _ = env.reset()
    agents = env.possible_agents

    obs_dim = env.observation_space(agents[0]).shape[0]
    act_dim = env.action_space(agents[0]).n

    print(f"Device  : {device}")
    print(f"Agents  : {agents}")
    print(f"Obs dim : {obs_dim}  |  Act dim: {act_dim}")
    print(f"Toplam timestep: {TOTAL_UPDATES * ROLLOUT_STEPS * N_AGENTS:,}\n")

    policy    = CommNetActorCritic(obs_dim, act_dim, N_AGENTS).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=TOTAL_UPDATES
    )

    reward_history       = []
    episodes_per_rollout = ROLLOUT_STEPS // MAX_CYCLES

    for update in range(TOTAL_UPDATES):
        buf = collect_rollout(env, policy, agents, device)
        ppo_update(policy, optimizer, buf, agents, device)
        scheduler.step()
        mean_reward = np.mean([
            sum(buf["per_agent_rewards"][a]) for a in agents
        ]) / episodes_per_rollout
        reward_history.append(mean_reward)

        if update % 100 == 0:
            print(f"Update {update:4d}/{TOTAL_UPDATES} | Mean Reward: {mean_reward:.3f}")

    torch.save(policy.state_dict(), "commnet_policy.pth")
    print("Model kaydedildi: commnet_policy.pth")

    plot_rewards(reward_history, N_AGENTS)


if __name__ == "__main__":
    main()
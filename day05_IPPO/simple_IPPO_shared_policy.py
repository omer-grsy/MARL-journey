import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpe2 import simple_spread_v3

# =========================
# 1. SHARED POLICY
# =========================
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.pi = nn.Linear(128, act_dim)
        self.v  = nn.Linear(128, 1)

    def forward(self, x):
        h     = self.net(x)
        dist  = torch.distributions.Categorical(logits=self.pi(h))
        value = self.v(h).squeeze(-1)
        return dist, value


# =========================
# 2. GAE
# =========================
def compute_gae(rewards, values, last_value, dones, gamma=0.99, lam=0.95):
    values_ext = np.append(values, last_value)
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv   = 0.0

    for t in reversed(range(T)):
        delta    = rewards[t] + gamma * values_ext[t+1] * (1 - dones[t]) - values_ext[t]
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns


# =========================
# 3. ROLLOUT
# =========================

def collect_rollout(env, policy, agents, rollout_length=1024):
    # Her ajan için ayrı buffer
    buffer = {a: {"obs": [], "actions": [], "rewards": [],
                  "dones": [], "log_probs": [], "values": []}
              for a in agents}

    obs, _ = env.reset()

    for _ in range(rollout_length):
        if not env.agents:
            obs, _ = env.reset()

        actions, log_probs, values = {}, {}, {}

        for agent in env.agents:
            o           = torch.tensor(obs[agent], dtype=torch.float32)
            dist, value = policy(o)  # shared policy
            action      = dist.sample()

            actions[agent]   = action.item()
            log_probs[agent] = dist.log_prob(action).item()
            values[agent]    = value.item()

        next_obs, rewards, terms, truncs, _ = env.step(actions)

        for agent in env.agents:
            done = terms.get(agent, False) or truncs.get(agent, False)
            buffer[agent]["obs"].append(obs[agent])
            buffer[agent]["actions"].append(actions[agent])
            buffer[agent]["rewards"].append(rewards.get(agent, 0.0))
            buffer[agent]["dones"].append(float(done))
            buffer[agent]["log_probs"].append(log_probs[agent])
            buffer[agent]["values"].append(values[agent])

        obs = next_obs if next_obs else {}

    # Her ajan için ayrı GAE, sonra birleştirilir
    all_obs, all_actions, all_lps = [], [], []
    all_adv, all_ret             = [], []

    for agent in agents:
        r = np.array(buffer[agent]["rewards"], dtype=np.float32)
        v = np.array(buffer[agent]["values"],  dtype=np.float32)
        d = np.array(buffer[agent]["dones"],   dtype=np.float32)

        if obs and agent in obs:
            with torch.no_grad():
                o_last      = torch.tensor(obs[agent], dtype=torch.float32)
                _, last_val = policy(o_last)
                last_value  = last_val.item()
        else:
            last_value = 0.0

        adv, ret = compute_gae(r, v, last_value, d)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        all_obs.append(np.array(buffer[agent]["obs"]))
        all_actions.append(np.array(buffer[agent]["actions"]))
        all_lps.append(np.array(buffer[agent]["log_probs"]))
        all_adv.append(adv)
        all_ret.append(ret)

    # GAE hesabından sonra concatenate
    combined = {
        "obs":        np.concatenate(all_obs),
        "actions":    np.concatenate(all_actions),
        "log_probs":  np.concatenate(all_lps),
        "advantages": np.concatenate(all_adv),
        "returns":    np.concatenate(all_ret),
        "per_agent_rewards": {a: buffer[a]["rewards"] for a in agents}
    }

    return combined


# =========================
# 4. PPO UPDATE
# =========================
def ppo_update(policy, optimizer, buf,
               ppo_epochs=10, mini_batch_size=256,
               clip_eps=0.2, value_coef=0.5, entropy_coef=0.05):
    """
    Shared policy olduğu için tüm ajanların verisi birleşik geliyor.
    3 ajan x 1024 adım = 3072 sample — daha zengin gradient.
    """
    obs        = torch.tensor(np.array(buf["obs"]),       dtype=torch.float32)
    actions    = torch.tensor(np.array(buf["actions"]),   dtype=torch.long)
    old_lps    = torch.tensor(np.array(buf["log_probs"]), dtype=torch.float32)
    advantages = torch.tensor(buf["advantages"],          dtype=torch.float32)
    returns    = torch.tensor(buf["returns"],             dtype=torch.float32)

    T = len(obs)  # 3 ajan x 1024 = 3072

    for _ in range(ppo_epochs):
        for idx in [np.random.permutation(T)[i:i + mini_batch_size]
                    for i in range(0, T, mini_batch_size)]:

            dist, values = policy(obs[idx])
            new_lps      = dist.log_prob(actions[idx])
            ratio        = torch.exp(new_lps - old_lps[idx])
            clipped      = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

            actor_loss  = -torch.min(ratio  * advantages[idx],
                                     clipped * advantages[idx]).mean()
            critic_loss = F.mse_loss(values, returns[idx])
            entropy     = dist.entropy().mean()

            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


# =========================
# 5. TRAINING LOOP
# =========================
N_AGENTS    = 3
N_UPDATES   = 300
MAX_CYCLES = 1024
ROLLOUT_LEN = 1024
LR          = 3e-4

env    = simple_spread_v3.parallel_env(N=N_AGENTS, max_cycles=MAX_CYCLES)
obs, _ = env.reset()
agents = env.possible_agents

obs_dim = env.observation_space(agents[0]).shape[0]
act_dim = env.action_space(agents[0]).n

print(f"Agents : {agents}")
print(f"Obs dim: {obs_dim}  |  Act dim: {act_dim}")
print(f"Shared policy — her update'te {N_AGENTS} x {ROLLOUT_LEN} = {N_AGENTS*ROLLOUT_LEN} sample\n")

# Tek policy, tek optimizer
policy    = Policy(obs_dim, act_dim)
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

reward_history = []

for update in range(N_UPDATES):
    buffer = collect_rollout(env, policy, agents, ROLLOUT_LEN)

    ppo_update(policy, optimizer, buffer)  # tek çağrı


    mean_reward = np.mean([
        sum(buffer["per_agent_rewards"][a]) for a in agents
    ])
    reward_history.append(mean_reward)

    if update % 20 == 0:
        print(f"Update {update:3d}/{N_UPDATES} | Mean Reward: {mean_reward:.3f}")

# =========================
# 6. PLOT
# =========================
plt.figure(figsize=(10, 4))
plt.plot(reward_history, alpha=0.4, label="Raw")

window   = 10
smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, len(reward_history)), smoothed,
         linewidth=2, label=f"Smoothed ({window})")

plt.xlabel("Update")
plt.ylabel("Mean Cooperative Reward")
plt.title(f"IPPO Shared Policy — simple_spread (N={N_AGENTS})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_curve_shared_n3.png", dpi=150)
plt.show()

print(f"\nİlk 10 update ort : {np.mean(reward_history[:10]):.3f}")
print(f"Son  10 update ort : {np.mean(reward_history[-10:]):.3f}")
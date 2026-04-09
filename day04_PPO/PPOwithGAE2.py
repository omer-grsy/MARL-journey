import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =========================
# NETWORKS
# =========================
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(x)


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)


# =========================
# SETUP
# =========================
env = gym.make("CartPole-v1")

policy_net = PolicyNetwork()
value_net = ValueNetwork()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=3e-4)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=3e-4)

gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2

batch_size = 2048
mini_batch_size = 64
ppo_epochs = 10

episode_rewards = []


# =========================
# GAE (FIXED)
# =========================
def compute_gae(rewards, values, dones):
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    return advantages


# =========================
# TRAINING
# =========================
obs, _ = env.reset()
total_steps = 0
episode_reward = 0  # ✅ FIX

while total_steps < 100000:

    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []

    # =========================
    # COLLECT DATA
    # =========================
    for _ in range(batch_size):

        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        logits = policy_net(obs_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = value_net(obs_tensor)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(obs_tensor)
        actions.append(action)
        log_probs.append(log_prob.detach())
        rewards.append(reward)
        values.append(value.item())
        dones.append(done)

        episode_reward += reward  # ✅ FIX

        obs = next_obs
        total_steps += 1

        if done:
            episode_rewards.append(episode_reward)  # ✅ FIX
            episode_reward = 0                      # ✅ RESET
            obs, _ = env.reset()

    # =========================
    # BOOTSTRAP VALUE
    # =========================
    with torch.no_grad():
        next_value = value_net(torch.tensor(obs, dtype=torch.float32)).item()
    values.append(next_value)

    # =========================
    # GAE
    # =========================
    advantages = compute_gae(rewards, values, dones)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states = torch.stack(states)
    actions = torch.tensor(actions)
    log_probs = torch.stack(log_probs)

    # =========================
    # PPO UPDATE
    # =========================
    for _ in range(ppo_epochs):

        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb_idx = indices[start:end]

            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            logits = policy_net(mb_states)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

            policy_loss = -torch.min(
                ratio * mb_advantages,
                clipped_ratio * mb_advantages
            ).mean()

            value_pred = value_net(mb_states).squeeze()
            value_loss = F.mse_loss(value_pred, mb_returns)

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            value_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            value_optimizer.step()

    if len(episode_rewards) > 0:
        print(f"Steps: {total_steps} | Last Reward: {episode_rewards[-1]}")

# =========================
# PLOT
# =========================
plt.plot(episode_rewards)
plt.title("Rewards")
plt.show()

print("Average last 100:", np.mean(episode_rewards[-100:]))

env.close()
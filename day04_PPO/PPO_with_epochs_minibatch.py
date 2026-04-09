import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =========================
#  POLICY (ACTOR)
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


# =========================
#  VALUE (CRITIC)
# =========================
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
#  SETUP
# =========================
env = gym.make("CartPole-v1")

policy_net = PolicyNetwork()
value_net = ValueNetwork()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=3e-4)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=3e-4)

clip_epsilon = 0.2
gamma = 0.99

episode_rewards = []

# =========================
# TRAINING LOOP
# =========================
for episode in range(300):

    obs, _ = env.reset()

    states = []
    actions = []
    log_probs_old = []
    rewards = []
    values = []

    done = False

    # -------- EPISODE --------
    while not done:

        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        logits = policy_net(obs_tensor)
        probs = F.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        log_prob = dist.log_prob(action)

        states.append(obs_tensor)
        actions.append(action)
        log_probs_old.append(log_prob.detach())
        values.append(value_net(obs_tensor))

        obs, reward, terminated, truncated, _ = env.step(action.item())

        rewards.append(reward)
        done = terminated or truncated

    # =========================
    # RETURNS
    # =========================
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.cat(values).squeeze()

    # =========================
    # ADVANTAGE
    # =========================
    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.detach()

    states = torch.stack(states)

    # =========================
    # PPO UPDATE
    # =========================
    for _ in range(10):   #

        policy_loss_terms = []

        for state, action, old_log_prob, adv in zip(states, actions, log_probs_old, advantages):

            logits = policy_net(state)
            probs = F.softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)
            new_log_prob = dist.log_prob(action)

            ratio = torch.exp(new_log_prob - old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

            entropy = dist.entropy()

            loss = -torch.min(ratio * adv, clipped_ratio * adv) - 0.01 * entropy

            policy_loss_terms.append(loss)

        policy_loss = torch.stack(policy_loss_terms).mean()

        # VALUE LOSS
        value_pred = value_net(states).squeeze()
        value_loss = 0.5 * F.mse_loss(value_pred, returns)

        # UPDATE
        optimizer.zero_grad()
        value_optimizer.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        optimizer.step()
        value_optimizer.step()

    # =========================
    # LOG
    # =========================
    total_reward = sum(rewards)
    episode_rewards.append(total_reward)

    print(f"Episode {episode} | Reward: {total_reward:.2f}")

# =========================
# PLOT
# =========================
plt.figure()
plt.plot(episode_rewards, label="Reward")
plt.title("Reward")
plt.legend()
plt.show()

avg_last_100 = np.mean(episode_rewards[-100:])
print(avg_last_100)
env.close()
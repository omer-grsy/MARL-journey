import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)  # 4 state → 2 action

    def forward(self, x):
        return self.fc(x)  # logits döner


#
env = gym.make("CartPole-v1")
net = PolicyNetwork()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

#
episode_rewards = []
moving_avg = []
entropies = []
losses = []

# --- Training ---
for episode in range(200):

    obs, _ = env.reset()

    log_probs = []
    rewards = []
    entropy_list = []

    done = False

    # Episode loop
    while not done:

        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        # --- Policy (π(a|s)) ---
        logits = net(obs_tensor)
        probs = F.softmax(logits, dim=0)

        dist = torch.distributions.Categorical(probs)

        # --- Action sample ---
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # --- Entropy (randomlık ölçüsü) ---
        entropy = dist.entropy()
        entropy_list.append(entropy)

        log_probs.append(log_prob)

        # --- Environment step ---
        obs, reward, terminated, truncated, _ = env.step(action.item())

        # --- Reward shaping ---
        pole_angle = obs[2]
        #reward = reward - 0.1 * abs(pole_angle)

        rewards.append(reward)

        done = terminated or truncated

    # --- Episode reward ---
    total_reward = sum(rewards)
    episode_rewards.append(total_reward)

    # --- Moving average (son 20 episode) ---
    avg_reward = sum(episode_rewards[-20:]) / len(episode_rewards[-20:])
    moving_avg.append(avg_reward)

    # --- Entropy ortalaması ---
    avg_entropy = torch.stack(entropy_list).mean().item()
    entropies.append(avg_entropy)

    #  REINFORCE learning ---

    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)

    # normalize (stabil learning için)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Loss hesapla
    loss_terms = []

    for log_prob, G in zip(log_probs, returns):
        loss_terms.append(-log_prob * G)

    loss = torch.stack(loss_terms).sum()

    # Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # --- Print ---
    print(f"Episode {episode} | Reward: {total_reward:.2f} | Avg(20): {avg_reward:.2f} | Entropy: {avg_entropy:.2f}")

env.close()


# --- PLOT ---

plt.figure(figsize=(15, 5))

# Reward
plt.subplot(1, 3, 1)
plt.plot(episode_rewards, label="Reward")
plt.plot(moving_avg, label="Avg(20)")
plt.title("Reward")
plt.legend()

# Entropy
plt.subplot(1, 3, 2)
plt.plot(entropies, label="Entropy")
plt.title("Entropy")
plt.legend()

# Loss
plt.subplot(1, 3, 3)
plt.plot(losses, label="Loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()
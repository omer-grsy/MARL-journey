import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- Policy Network ---
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)  # 4 state → 2 action

    def forward(self, x):
        return self.fc(x)  # logits döner

class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 1)
    def forward(self, x):
        return self.fc1(x)

# --- Setup ---
env = gym.make("CartPole-v1")
policy_net = PolicyNetwork()
value_net = ValueNetwork()
policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=0.01)

# --- Tracking ---
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
    values =[]

    done = False

    #Episode loop
    while not done:

        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        # --- Policy (π(a|s)) ---
        logits = policy_net(obs_tensor)
        probs = F.softmax(logits, dim=0)

        dist = torch.distributions.Categorical(probs)

        # --- Action sample ---
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)

        ### Value(state değeri tahmini)
        value = value_net(obs_tensor)
        values.append(value)

        # --- Entropy (randomlık ölçüsü) ---
        entropy = dist.entropy()
        entropy_list.append(entropy)



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


    #--- REINFORCE learning ---

    # Returns hesapla
    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.cat(values).squeeze()
    advantages = returns - values.detach()

    # normalize (stabil learning için)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Loss hesapla
    policy_loss_terms = []

    for log_prob, adv  in zip(log_probs, advantages):
        policy_loss_terms.append(-log_prob * adv)

    policy_loss = torch.stack(policy_loss_terms).sum()

    value_loss = F.mse_loss(values, returns)

    # Update
    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    policy_loss.backward()
    value_loss.backward()
    policy_optimizer.step()
    value_optimizer.step()
    #losses.append(loss.item())

    # --- Print ---
    print(f"Episode {episode} | Reward: {total_reward:.2f} | Avg(20): {avg_reward:.2f} | Entropy: {avg_entropy:.2f}")

env.close()


# --- PLOT ---

plt.figure(figsize=(15, 5))

# Reward
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, label="Reward")
plt.plot(moving_avg, label="Avg(20)")
plt.title("Reward")
plt.legend()

# Entropy
plt.subplot(1, 2, 2)
plt.plot(entropies, label="Entropy")
plt.title("Entropy")
plt.legend()

# Loss
# plt.subplot(1, 3, 3)
# plt.plot(losses, label="Loss")
# plt.title("Loss")
# plt.legend()

plt.tight_layout()
plt.show()
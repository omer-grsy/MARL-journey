import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#Policy Network ---
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)  # 4 state → 2 action (sol / sağ)
        self.fc2 = nn.Linear(64, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x)) # fc1 aktivasyon
        return self.fc2(x)      # fc2 final output

# TRAIN FUNCTION
def train(shaping=False):

    env = gym.make("CartPole-v1")
    net = PolicyNetwork()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    episode_rewards = []  # her episode toplam reward

    for episode in range(200):

        obs, _ = env.reset()  # environment reset → initial state

        log_probs = []  # seçilen action'ların log prob'ları
        rewards = []    # her step reward

        done = False

        # EPISODE LOOP
        while not done:

            obs_tensor = torch.tensor(obs, dtype=torch.float32)  # numpy → tensor

            logits = net(obs_tensor)        # model → raw scores
            probs = F.softmax(logits, dim=0)  # logits → probability  policy--> net + softmax

            dist = torch.distributions.Categorical(probs)  # action distribution
            action = dist.sample()                         # action seç (random)
            log_prob = dist.log_prob(action)               # seçilen action'ın log prob'u

            log_probs.append(log_prob)  # learning için sakla

            obs, reward, terminated, truncated, _ = env.step(action.item())
            # action uygula → yeni state + reward

            # reward shaping
            if shaping:
                pole_angle = obs[2]             # state içinden angle al
                #reward = reward - 0.1 * abs(pole_angle)  # dik durmaya teşvik
                reward = reward + 0.1 * (0.2 - abs(pole_angle))  # alternatif
            rewards.append(reward)

            done = terminated or truncated

        # --- EPISODE BİTTİ ---
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # --- RETURNS (gelecek reward toplamı) ---
        returns = []
        G = 0
        gamma = 0.99 # Discount factor
        for r in reversed(rewards):  # sondan başa git
            G = r + gamma * G               # future reward ekle
            returns.insert(0, G)    # başa ekle (doğru sırayı koru)

        returns = torch.tensor(returns, dtype=torch.float32)

        #normalize (learning stabil olsun)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # --- LOSS (REINFORCE) ---
        loss_terms = []

        for log_prob, G in zip(log_probs, returns):
            loss_terms.append(-log_prob * G)
            # iyi action → prob artır
            # kötü action → prob azalt

        loss = torch.stack(loss_terms).sum()  # tüm step'leri topla

        # --- UPDATE ---
        optimizer.zero_grad()  # gradient sıfırla
        loss.backward()        # gradient hesapla
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
        optimizer.step()       # ağırlıkları güncelle

    env.close()
    return episode_rewards  # grafikte kullanacağız

# --- MOVING AVERAGE ---
def moving_avg(data, window=10):
    result = []
    for i in range(len(data)):
        if i >= window:
            avg = sum(data[i-window:i]) / window  # son N episode ortalaması
            result.append(avg)
        else:
            result.append(None)  # yeterli veri yok
    return result

# RUN 1: SHAPING YOK
rewards_no = train(shaping=False)

# RUN 2: SHAPING VAR
rewards_shape = train(shaping=True)

# --- PLOT ---
plt.figure()

plt.plot(moving_avg(rewards_no), label="No Shaping")      # normal reward
plt.plot(moving_avg(rewards_shape), label="With Shaping") # değiştirilmiş reward

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Shaping vs No Shaping")
plt.legend()

plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class PolicyGradientAgent:
    def __init__(self, env, gamma=0.99, lr=0.01, episodes=500):
        self.env = env
        self.gamma = gamma
        self.policy_net = PolicyNetwork(2, 4)  # 2 input features (x, y)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.episodes = episodes
        self.episode_rewards = []
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    def train(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            rewards, log_probs = [], []
            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                probs = self.policy_net(state_tensor)
                action_idx = torch.multinomial(probs, 1).item()
                log_probs.append(torch.log(probs[action_idx]))
                
                action = self.actions[action_idx]
                next_state, reward, done = self.env.step(action)
                rewards.append(reward)
                state = next_state

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)

            loss = -sum(log_prob * G for log_prob, G in zip(log_probs, returns))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.episode_rewards.append(sum(rewards))

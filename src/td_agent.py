import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, episodes=500, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.env = env
        self.q_table = np.zeros((env.grid_size, env.grid_size, 4))  # Q-table
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.episode_rewards = []

    def choose_action(self, state):
        x, y = state
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        return self.actions[np.argmax(self.q_table[x, y])]

    def train(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                x, y = state
                nx, ny = next_state
                action_idx = self.actions.index(action)
                
                # Q-value update
                best_next_q = np.max(self.q_table[nx, ny])
                self.q_table[x, y, action_idx] += self.alpha * (
                    reward + self.gamma * best_next_q - self.q_table[x, y, action_idx]
                )
                
                state = next_state
                total_reward += reward

            self.episode_rewards.append(total_reward)
            self.epsilon *= 0.99  # Decay epsilon

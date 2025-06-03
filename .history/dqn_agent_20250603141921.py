import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_net = QNetwork(obs_dim, act_dim)
        self.target_net = QNetwork(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, obs):
        if random.random() < self.epsilon:
            return np.random.randint(self.act_dim)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return torch.argmax(q_values).item()

    def train_step(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return

        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)

        obs = torch.FloatTensor(obs)
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_obs = torch.FloatTensor(next_obs)
        done = torch.FloatTensor(done).unsqueeze(1)

        q_val = self.q_net(obs).gather(1, action)
        with torch.no_grad():
            max_next_q = self.target_net(next_obs).max(1, keepdim=True)[0]
            target = reward + self.gamma * max_next_q * (1 - done)

        loss = nn.functional.mse_loss(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

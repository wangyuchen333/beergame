import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def add(self, obs, action, reward, next_obs, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = map(np.array, zip(*samples))
        return obs, action, reward, next_obs, done

    def __len__(self):
        return len(self.buffer)

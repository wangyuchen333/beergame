import gym
import numpy as np

class BeerGameEnv(gym.Env):
    def __init__(self, agent_index=0, delay=1, max_order=20, max_steps=50):
        super().__init__()
        self.num_agents = 3
        self.agent_index = agent_index
        self.max_order = max_order
        self.max_steps = max_steps
        self.delay = delay

        self.observation_space = gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(max_order + 1)

        self.reset()

    def reset(self):
        self.time = 0
        self.inventory = [20] * self.num_agents
        self.backlog = [0] * self.num_agents
        self.in_transit = [[0] * self.delay for _ in range(self.num_agents)]
        self.last_demand = [0] * self.num_agents
        return self._get_obs(self.agent_index)

    def _get_obs(self, i):
        return np.array([
            self.inventory[i],
            self.backlog[i],
            self.last_demand[i]
        ], dtype=np.float32)

    def _fixed_policy(self, idx):
        return 4  # 固定订购量

    def step(self, action):
        orders = []
        for i in range(self.num_agents):
            if i == self.agent_index:
                orders.append(action)
            else:
                orders.append(self._fixed_policy(i))

        deliveries = [self.in_transit[i].pop(0) for i in range(self.num_agents)]
        for i in range(self.num_agents):
            self.in_transit[i].append(orders[i])

        # 零售商客户需求（泊松分布）
        customer_demand = np.random.poisson(4)
        self.last_demand[0] = customer_demand

        for i in range(self.num_agents):
            demand = customer_demand if i == 0 else orders[i - 1]
            shipped = min(self.inventory[i], demand)
            self.inventory[i] -= shipped
            self.backlog[i] = max(0, self.backlog[i] + demand - shipped)

        idx = self.agent_index
        reward = - (0.5 * self.inventory[idx] + 1.0 * self.backlog[idx])

        self.time += 1
        done = self.time >= self.max_steps
        obs = self._get_obs(idx)
        return obs, reward, done, {}

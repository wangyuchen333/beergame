# beergame_env.py（核心环境逻辑）
import gym
import numpy as np

class SimpleBeerGame(gym.Env):
    def __init__(self, demand_dist=None, max_order=20, delay=1, train_agent_index=0):
        super().__init__()
        self.num_agents = 4  # 零售、批发、分销、工厂
        self.max_order = max_order
        self.delay = delay
        self.train_agent_index = train_agent_index

        # 状态：库存、积压、上周期需求
        self.observation_space = gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(max_order + 1)

        self.reset()

    def reset(self):
        self.inventory = [20] * self.num_agents
        self.backlog = [0] * self.num_agents
        self.in_transit = [[0]*self.delay for _ in range(self.num_agents)]
        self.last_demand = [0] * self.num_agents
        self.time = 0
        self.done = False
        return self._get_obs(self.train_agent_index)

    def _get_obs(self, i):
        return np.array([self.inventory[i], self.backlog[i], self.last_demand[i]], dtype=np.float32)

    def step(self, action):
        orders = []
        for i in range(self.num_agents):
            if i == self.train_agent_index:
                orders.append(action)
            else:
                orders.append(self._random_policy(i))

        # Demand from outside world (only for retailer)
        customer_demand = np.random.poisson(4)
        self.last_demand[0] = customer_demand

        deliveries = [0] * self.num_agents
        for i in range(self.num_agents):
            deliveries[i] = self.in_transit[i].pop(0)
            self.in_transit[i].append(orders[i])

        # Order fulfillment and state update
        for i in range(self.num_agents):
            total_demand = customer_demand if i == 0 else orders[i - 1]
            shipped = min(self.inventory[i], total_demand)
            self.inventory[i] -= shipped
            backlog_change = total_demand - shipped
            self.backlog[i] += max(0, backlog_change)

        # Update reward (only for train agent)
        idx = self.train_agent_index
        holding_cost = self.inventory[idx]
        backlog_cost = self.backlog[idx]
        reward = - (0.5 * holding_cost + 1.0 * backlog_cost)

        self.time += 1
        if self.time >= 50:
            self.done = True

        obs = self._get_obs(idx)
        return obs, reward, self.done, {}

    def _random_policy(self, idx):
        return np.random.randint(0, self.max_order + 1)

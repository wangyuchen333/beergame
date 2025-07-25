import gym
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from utils import ReplayBuffer
import numpy as np

# 环境配置
ENV_NAME = "BeerGame-v0"  # 可替换为你自定义的
NUM_EPISODES = 500
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10

def train():
    from beergame_env import SimpleBeerGame

    env = SimpleBeerGame(train_agent_index=0)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = DQNAgent(obs_dim, act_dim)
    agent = DQNAgent(obs_dim, act_dim)
    buffer = ReplayBuffer()

    rewards = []

    for ep in range(NUM_EPISODES):
        obs = env.reset()
        ep_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

            agent.train_step(buffer, BATCH_SIZE)
            ep_reward += reward

        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        rewards.append(ep_reward)
        print(f"Episode {ep}, Reward: {ep_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    # 绘图
    plt.plot(rewards)
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.savefig("reward_curve.png")
    plt.show()

    env.close()

if __name__ == "__main__":
    train()

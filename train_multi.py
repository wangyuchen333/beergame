import numpy as np
from dqn import DQNAgent, test_agent
from env import Env
import matplotlib.pyplot as plt
import os
import torch
torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 或 'STHeiti', 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False  # 绘图显示负号

    # 创建保存模型和图表的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # 环境参数
    num_firms = 3
    p = [10, 9, 8]
    h = 0.5
    c = 2
    initial_inventory = 100
    poisson_lambda = 10
    max_steps = 100

    # 创建环境
    env = Env(num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps)

    # 为每个企业创建一个 DQN 智能体
    agents = []
    for firm_id in range(num_firms):
        state_size = 3  # 订单、满足的需求、库存
        action_size = 20  # 最大订单量为20
        agent = DQNAgent(state_size=state_size, action_size=action_size, firm_id=firm_id, max_order=action_size, buffer_size=20000, batch_size=64, gamma=0.99, learning_rate=5e-4, tau=1e-3, update_every=2)
        agents.append(agent)

    # 训练
    num_episodes = 2000
    scores = np.zeros((num_firms, num_episodes))

    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = np.zeros((num_firms, 1))
        done = False

        while not done:
            actions = np.zeros((num_firms, 1))
            for firm_id, agent in enumerate(agents):
                # 每个智能体选择动作
                actions[firm_id] = agent.act(state[firm_id].reshape(1, -1), epsilon=0.1)

            # 环境更新
            next_state, rewards, done = env.step(actions)

            # 每个智能体学习
            for firm_id, agent in enumerate(agents):
                agent.step(state[firm_id].reshape(1, -1), actions[firm_id], rewards[firm_id], next_state[firm_id].reshape(1, -1), done)
                episode_rewards[firm_id] += rewards[firm_id]

            state = next_state

        # 记录每个智能体的总奖励（修复警告）
        for firm_id in range(num_firms):
            scores[firm_id, episode] = float(episode_rewards[firm_id])  # 转换为标量

        if episode % 100 == 0:
            print(f"Episode {episode}, Average Rewards: {np.mean(episode_rewards):.2f}, Reward: {episode_rewards}")

    # 绘制每个智能体的训练曲线
    for firm_id in range(num_firms):
        plt.figure(figsize=(10, 6))
        plt.plot(scores[firm_id], alpha=0.3, label='原始奖励')
        # 计算移动平均
        window_size = 100
        moving_avg = np.convolve(scores[firm_id], np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(scores[firm_id])), moving_avg, label=f'{window_size}个episode的移动平均')
        plt.title(f'企业 {firm_id} 的训练奖励')
        plt.xlabel('Episode')
        plt.ylabel('奖励')
        plt.legend()
        plt.savefig(f'figures/training_rewards_firm_{firm_id}.png')
        plt.close()

    # 测试所有智能体
    for firm_id, agent in enumerate(agents):
        test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history = test_agent(env, agent, num_episodes=10)
        
        # 绘制测试结果
        plt.figure(figsize=(14, 10))
        
        # 库存图表
        plt.subplot(2, 2, 1)
        plt.plot(np.mean(inventory_history, axis=0))
        plt.title(f'企业 {firm_id} 的平均库存')
        plt.xlabel('时间步')
        plt.ylabel('库存量')
        
        # 订单图表
        plt.subplot(2, 2, 2)
        plt.plot(np.mean(orders_history, axis=0))
        plt.title(f'企业 {firm_id} 的平均订单量')
        plt.xlabel('时间步')
        plt.ylabel('订单量')
        
        # 需求和满足需求图表
        plt.subplot(2, 2, 3)
        plt.plot(np.mean(demand_history, axis=0), label='需求')
        plt.plot(np.mean(satisfied_demand_history, axis=0), label='满足的需求')
        plt.title(f'企业 {firm_id} 的平均需求 vs 满足的需求')
        plt.xlabel('时间步')
        plt.ylabel('数量')
        plt.legend()
        
        # 奖励柱状图
        plt.subplot(2, 2, 4)
        plt.bar(range(len(test_scores)), test_scores)
        plt.title(f'企业 {firm_id} 的测试episode奖励')
        plt.xlabel('Episode')
        plt.ylabel('总奖励')
        
        plt.tight_layout()
        plt.savefig(f'figures/test_results_firm_{firm_id}.png')
        plt.close()

        # 保存模型
        agent.save(f'models/dqn_agent_firm_{firm_id}_final.pth')
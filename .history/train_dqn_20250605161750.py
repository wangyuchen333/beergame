import numpy as np
from dqn import DQNAgent, train_dqn, plot_training_results, test_agent, plot_test_results
from env import Env

if __name__ == "__main__":
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

    # 智能体参数
    firm_id = 1  # 训练第2个企业
    state_size = 3  # 订单、满足的需求、库存
    action_size = 20  # 最大订单量为20

    agent = DQNAgent(state_size=state_size, action_size=action_size, firm_id=firm_id, max_order=action_size)

    # 训练
    scores = train_dqn(env, agent, num_episodes=2000, max_t=max_steps, eps_start=1.0, eps_end=0.01, eps_decay=0.995)

    # 绘制训练曲线
    plot_training_results(scores)

    # 测试
    test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history = test_agent(env, agent, num_episodes=10)
    plot_test_results(test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history)
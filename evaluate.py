import rlcard
import torch
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.agents.random_agent import RandomAgent
import numpy as np

# 加载环境
env = rlcard.make('no-limit-holdem', config={'seed': 42})
num_actions = env.num_actions
state_shape = env.state_shape[0]

# 创建 DQN agent 并加载训练好的模型
agent = DQNAgent(
    num_actions=num_actions,
    state_shape=state_shape,
    mlp_layers=[128, 128]
)
agent._q.load_state_dict(torch.load('./checkpoints/dqn_episode_10000.pth'))  # 改成你的模型路径
agent._q.eval()

# 创建 RandomAgent
random_agent = RandomAgent(num_actions=num_actions)

# 设置 agent 对战：agent0 vs random
env.set_agents([agent, random_agent])

# 模拟 20000 局，累计收益
num_games = 20000
total_return = 0

for _ in range(num_games):
    _, payoffs = env.run(is_training=False)
    total_return += payoffs[0]  # DQN agent 的收益

# 计算 bb/100
bb_per_100 = (total_return / num_games) * 100
print(f'[DQN vs Random] bb/100 over {num_games} hands: {bb_per_100:.2f}')

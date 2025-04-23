import rlcard
import torch
from rlcard.agents.random_agent import RandomAgent

# 加载环境
env = rlcard.make('no-limit-holdem', config={'seed': 42})

# 直接加载完整 agent
agent = torch.load('agent_full.pth')  # ← 你保存的完整对象

# 创建随机 agent
random_agent = RandomAgent(num_actions=env.num_actions)

# 设置 agent 对战（DQN vs Random）
env.set_agents([agent, random_agent])

# 模拟 20000 局
num_games = 20000
total_return = 0

for e in range(num_games):
    _, payoffs = env.run(is_training=False)
    total_return += payoffs[0]  # DQN agent 的收益

    if e % 500 == 0:
        print(total_return)

# 计算 bb/100
bb_per_100 = (total_return / num_games) * 100
print(f'[DQN vs Random] bb/100 over {num_games} hands: {bb_per_100:.2f}')

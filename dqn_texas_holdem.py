import rlcard
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.utils import tournament
import random
import numpy as np

# 设置随机种子确保可复现性
random.seed(42)
np.random.seed(42)

# 初始化环境
env = rlcard.make('no-limit-holdem', config={'seed': 42, 'allow_step_back': True})
eval_env = rlcard.make('no-limit-holdem', config={'seed': 42})

# 初始化 DQN Agent
agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[128, 128],
    replay_memory_size=10000,
    train_every=1,
    save_every=100,
    epsilon_decay_steps=10000,
    batch_size=32
)

# 设置 agent
env.set_agents([agent, agent])  # self-play
eval_env.set_agents([agent, agent])

# 训练 agent
episode_num = 10000
for episode in range(episode_num):
    trajectories, _ = env.run(is_training=True)
    
    trajectory = trajectories[0]  # 一个玩家的完整轨迹
    for i in range(0, len(trajectory) - 2, 2):  # 每两个元素是一组
        state = trajectory[i]
        reward = trajectory[i + 1]
        next_state = trajectory[i + 2]
        done = (i + 2 >= len(trajectory) - 1)

        # 获取动作（从状态中取出）
        action = state.get('action', None)  # 有些版本中 'action' 可能不存在

        if action is None:
            continue  # 跳过无效记录（例如游戏刚开始前的空状态）

        legal_actions = list(next_state['legal_actions'].keys())
        agent.feed_memory(state['obs'], action, reward, next_state['obs'], legal_actions, done)

    # 每隔一段评估一次
    if episode % 500 == 0:
        reward = tournament(eval_env, 100)[0]
        print(f'Episode {episode}, Eval win rate: {reward:.3f}')



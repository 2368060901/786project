import rlcard
import random
from rlcard.agents.random_agent import RandomAgent
from ppo_agent import PPOAgent
from rlcard.utils import tournament
import torch
import numpy as np
import os

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Set up environment
env = rlcard.make('no-limit-holdem')
eval_env = rlcard.make('no-limit-holdem')

# PPO agent with small network
ppo_agent = PPOAgent(
    action_num=env.num_actions,
    state_shape=env.state_shape[0],  # ✅ 注意是 [0]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    lr=3e-4
)


# Set PPO as player 0, random agent as opponent
env.set_agents([ppo_agent, ppo_agent])
eval_env.set_agents([ppo_agent, ppo_agent])

# Train PPO agent
for episode in range(100000):
    trajectories, _ = env.run(is_training=True)
    trajectory = trajectories[0]
    for i in range(0, len(trajectory) - 2, 2):
        state = trajectory[i]
        reward = trajectory[i + 1]
        next_state = trajectory[i + 2]
        done = (i + 2 >= len(trajectory) - 1)

        # 你手动构造一个 transition dict
        transition = {
            'obs': state['obs'],
            'action': state['action'],  # 确保 action 是你在 step() 里存进去的
            'reward': reward,
            'next_state': next_state,
            'done': done
        }

        ppo_agent.feed(transition)

    if episode % 1000 == 0:
        reward = tournament(eval_env, 100)[0]
        print(f"Episode {episode}, PPO vs Random Win Rate: {reward:.3f}")

# Final evaluation
final_reward = tournament(eval_env, 500)[0]
print(f"Final PPO vs Random Win Rate: {final_reward:.3f}")

torch.save(ppo_agent.model.state_dict(), './ppo_model.pth')

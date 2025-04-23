import rlcard
import torch
from itertools import combinations
from tqdm import tqdm
from rlcard.agents.random_agent import RandomAgent
from ppo_agent import PPOAgent
import numpy as np

class AllCallsAgent:
    def __init__(self):
        self.use_raw = False
    def step(self, state):
        # 总是 CALL（动作1），如果不合法则选第一个合法动作
        return 1 if 1 in state['legal_actions'] else list(state['legal_actions'].keys())[0]
    def eval_step(self, state):
        return self.step(state), []

class AllRaisesAgent:
    def __init__(self):
        self.use_raw = False
    def step(self, state):
        return 2 if 2 in state['legal_actions'] else list(state['legal_actions'].keys())[0]
    def eval_step(self, state):
        return self.step(state), []

class AllFoldsAgent:
    def __init__(self):
        self.use_raw = False
    def step(self, state):
        return 0 if 0 in state['legal_actions'] else list(state['legal_actions'].keys())[0]
    def eval_step(self, state):
        return self.step(state), []
    
class AllRandomAgent:
    def __init__(self): self.use_raw = False
    def step(self, state): return np.random.choice(list(state['legal_actions'].keys()))
    def eval_step(self, state): return self.step(state), []

# 加载你的 DQN agent（完整保存）
# rl_agent = torch.load('agent_full.pth')
env = rlcard.make('no-limit-holdem')
state_shape = env.state_shape[0]
action_num = env.num_actions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

rl_agent = PPOAgent(action_num=action_num, state_shape=state_shape)
rl_agent.model.load_state_dict(torch.load('ppo_model.pth')) 
rl_agent.model.eval()

# 对手列表
agents = {
    'RLModule': rl_agent,
    # 'AllCalls': AllCallsAgent(),
    # 'AllRaises': AllRaisesAgent(),
    # 'AllFolds': AllFoldsAgent(),
    'AllRandom': AllRandomAgent()
}

# 模拟每种对战
num_games = 20000

# 结果记录：每个玩家对每个玩家的 bb/100
results = {name: {} for name in agents}

# 所有两两组合（不重复）
for name1, name2 in combinations(agents.keys(), 2):
    env = rlcard.make('no-limit-holdem', config={'seed': 42})
    env.set_agents([agents[name1], agents[name2]])

    total_1 = 0
    for _ in tqdm(range(num_games), desc=f'{name1} vs {name2}'):
        _, payoffs = env.run(is_training=False)
        total_1 += payoffs[0]

    bb_per_100 = (total_1 / num_games) * 100
    results[name1][name2] = round(bb_per_100, 2)
    results[name2][name1] = round(-bb_per_100, 2)

# 输出结果表格
print("\n🏆 BB/100 小组循环赛结果（正数表示行胜列）：\n")
players = list(agents.keys())
header = "".join([f"{p:>10}" for p in [""] + players])
print(header)

for p1 in players:
    row = f"{p1:>10}"
    for p2 in players:
        if p1 == p2:
            row += f"{'--':>10}"
        else:
            val = results[p1].get(p2, "--")
            row += f"{val:>10}"
    print(row)
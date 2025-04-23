import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64]):
        super().__init__()
        layers = []
        dims = [state_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        self.body = nn.Sequential(*layers)
        self.action_head = nn.Linear(dims[-1], action_dim)
        self.value_head = nn.Linear(dims[-1], 1)

    def forward(self, x):
        x = self.body(x)
        return self.action_head(x), self.value_head(x)

class PPOAgent:
    def __init__(self, action_num, state_shape, device=None, gamma=0.99, clip_eps=0.2, lr=3e-4):
        self.state_dim = state_shape[0]
        self.action_dim = action_num
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_raw = False

        self.model = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []

    def step(self, state):
        obs = torch.tensor(state['obs'], dtype=torch.float32).to(self.device).unsqueeze(0)
        logits, _ = self.model(obs)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        legal_actions = list(state['legal_actions'].keys())

        # 掩码非法动作
        probs_masked = np.zeros_like(probs)
        probs_masked[legal_actions] = probs[legal_actions]

        # 归一化
        sum_probs = probs_masked.sum()
        if sum_probs == 0:
            # fallback: uniform over legal actions
            for a in legal_actions:
                probs_masked[a] = 1.0
            probs_masked /= probs_masked.sum()
        else:
            probs_masked /= sum_probs

        action = np.random.choice(len(probs_masked), p=probs_masked)
        state['action'] = action
        return action

    def feed(self, transition):
        self.memory.append(transition)
        if len(self.memory) >= 256:
            self.train()
            self.memory = []

    def train(self):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for ts in self.memory:
            s = ts['obs']
            a = ts['action']
            r = ts['reward']
            ns = ts['next_state']['obs'] if ts['next_state'] else s
            d = ts['done']
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        logits, state_values = self.model(states)
        dists = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            returns = []
            ret = 0
            for r, d in zip(reversed(rewards), reversed(dones)):
                ret = r + self.gamma * ret * (1 - d)
                returns.insert(0, ret)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            advantages = returns - state_values.squeeze(1)

        # PPO loss
        old_log_probs = action_log_probs.detach()
        for _ in range(4):
            logits, state_values = self.model(states)
            log_probs = F.log_softmax(logits, dim=-1)
            new_action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(new_action_log_probs - old_log_probs)
            clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            loss = -torch.min(ratio * advantages, clip_adv).mean() + F.mse_loss(state_values.squeeze(1), returns)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval_step(self, state):
        action = self.step(state)
        return action, {}  # 第二个是 info（空字典即可）


    def set_device(self, device):
        self.device = device
        self.model.to(self.device)

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import random
from collections import deque
class TradingEnv:
    def __init__(self, prices):
        self.prices = prices
    def reset(self):
        self.i, self.pos, self.buy = 0, 0, 0
        return np.array([self.prices[self.i], self.pos])
    def step(self, action):
        price = self.prices[self.i]
        reward = 0
        if action == 1 and self.pos == 0:      
            self.pos, self.buy = 1, price

        elif action == 2 and self.pos == 1:    
            reward = price - self.buy
            self.pos = 0

        self.i += 1
        done = self.i == len(self.prices) - 1
        next_state = np.array([self.prices[self.i], self.pos])
        return next_state, reward, done

class DQN(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, a)
        )

    def forward(self, x): return self.net(x)

class DDQN:
    def __init__(self, s_dim, a_dim):
        self.gamma, self.epsilon, self.eps_min, self.eps_decay = 0.95, 1.0, 0.05, 0.995
        self.buffer = deque(maxlen=3000)

        self.policy = DQN(s_dim, a_dim)
        self.target = DQN(s_dim, a_dim)
        self.target.load_state_dict(self.policy.state_dict())

        self.opt = optim.Adam(self.policy.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.a_dim = a_dim

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.a_dim)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.policy(state).argmax().item()

    def train(self, batch=32):
        if len(self.buffer) < batch: return
        samp = random.sample(self.buffer, batch)

        s = torch.tensor([x[0] for x in samp], dtype=torch.float32)
        a = torch.tensor([x[1] for x in samp])
        r = torch.tensor([x[2] for x in samp], dtype=torch.float32)
        s2 = torch.tensor([x[3] for x in samp], dtype=torch.float32)
        d = torch.tensor([x[4] for x in samp], dtype=torch.float32)

        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze()
        next_act = self.policy(s2).argmax(dim=1)
        q_next = self.target(s2).gather(1, next_act.unsqueeze(1)).squeeze()
        target = r + (1 - d) * self.gamma * q_next

        loss = self.loss_fn(q, target.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

 
if __name__ == "__main__":
    prices = np.sin(np.linspace(0, 20, 500)) * 20 + 100
    env, agent = TradingEnv(prices), DDQN(2, 3)

    for ep in range(30):
        s, done, profit = env.reset(), False, 0
        while not done:
            a = agent.act(s)
            s2, r, done = env.step(a)
            agent.buffer.append((s, a, r, s2, done))
            s = s2
            profit += r
            agent.train()

        agent.update_target()
        print(f"Episode {ep+1}: Profit = {profit:.2f}")

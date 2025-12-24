import numpy as np

# =========================================================
# PART 1: FINANCIAL PORTFOLIO MANAGEMENT (ACTOR–CRITIC)
# =========================================================

class PortfolioEnv:
    def _init_(self, prices):
        self.prices = prices
        self.reset()

    def reset(self):
        self.t = 0
        self.portfolio_value = 1.0
        return self.prices[self.t]

    def step_env(self, action):
        self.t += 1
        price_diff = self.prices[self.t] - self.prices[self.t - 1]

        # Reward = profit - risk penalty
        reward = action * price_diff
        risk = 0.01 * np.var(self.prices[:self.t + 1])
        reward -= risk

        self.portfolio_value += reward
        return self.prices[self.t], reward


class Actor:
    def act(self, state):
        return np.tanh(state / 100)   # investment action


class Critic:
    def value(self, state):
        return state / 100


def run_portfolio_actor_critic():
    prices = np.array([100, 102, 101, 105, 108, 110])
    env = PortfolioEnv(prices)

    actor = Actor()
    critic = Critic()
    gamma = 0.9

    state = env.reset()

    print("\n===============================")
    print("ACTOR–CRITIC PORTFOLIO MANAGEMENT")
    print("===============================\n")

    for step in range(len(prices) - 1):
        action = actor.act(state)
        next_state, reward = env.step_env(action)

        advantage = reward + gamma * critic.value(next_state) - critic.value(state)

        print(f"Step {step+1} | Reward: {reward:.3f} | Advantage: {advantage:.3f}")
        state = next_state

    print("\nFinal Portfolio Value:", env.portfolio_value)


# =========================================================
# PART 2: ROBOT MAZE NAVIGATION (REINFORCE)
# =========================================================

maze = [
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 0, 2]   # 2 = Goal
]

actions = [(-1,0), (1,0), (0,-1), (0,1)]
alpha = 0.01
gamma = 0.9

def reward(cell):
    if cell == 2:
        return 10
    elif cell == 1:
        return -5
    else:
        return -1

theta = np.random.rand(4)

def choose_action():
    probs = np.exp(theta) / np.sum(np.exp(theta))
    return np.random.choice(4, p=probs)


def run_maze_reinforce():
    print("\n===============================")
    print("ROBOT MAZE NAVIGATION (REINFORCE)")
    print("===============================\n")

    global theta
    for episode in range(200):
        x, y = 0, 0
        trajectory = []

        while maze[x][y] != 2:
            action = choose_action()
            nx, ny = x + actions[action][0], y + actions[action][1]

            if 0 <= nx < 3 and 0 <= ny < 4:
                x, y = nx, ny

            r = reward(maze[x][y])
            trajectory.append((action, r))

            if maze[x][y] == 2:
                break

        G = 0
        for action, r in reversed(trajectory):
            G = r + gamma * G
            theta[action] += alpha * G

    print("Training Completed")
    print("Final Policy Parameters:", theta)


# =========================================================
# MAIN
# =========================================================

if __name__ == "_main_":
    run_portfolio_actor_critic()
    run_maze_reinforce()
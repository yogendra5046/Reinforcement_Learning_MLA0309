import numpy as np
import random

# ---------------- ENVIRONMENT SETUP ----------------
states = ["NORMAL", "HUMAN_PRESENT", "EMERGENCY"]
actions = ["SAFE_ACTION", "RISKY_ACTION", "UNETHICAL_ACTION"]

state_size = len(states)
action_size = len(actions)

# Q-table initialization
Q = np.zeros((state_size, action_size))

# Hyperparameters
alpha = 0.1        # learning rate
gamma = 0.9        # discount factor
epsilon = 0.2      # exploration rate
episodes = 500

harmful_actions = 0

# ---------------- REWARD FUNCTION (ETHICAL) ----------------
def get_reward(state, action):
    if action == "SAFE_ACTION":
        return 10
    if action == "RISKY_ACTION":
        return -2
    if action == "UNETHICAL_ACTION":
        if state == "HUMAN_PRESENT":
            return -20   # heavy penalty for harming humans
        return -5
    return 0

# ---------------- TRAINING ----------------
for _ in range(episodes):
    state = random.randint(0, state_size - 1)

    if random.uniform(0, 1) < epsilon:
        action = random.randint(0, action_size - 1)
    else:
        action = np.argmax(Q[state])

    reward = get_reward(states[state], actions[action])

    if actions[action] == "UNETHICAL_ACTION" and states[state] == "HUMAN_PRESENT":
        harmful_actions += 1

    next_state = random.randint(0, state_size - 1)

    Q[state, action] = Q[state, action] + alpha * (
        reward + gamma * np.max(Q[next_state]) - Q[state, action]
    )

# ---------------- EVALUATION ----------------
print("Learned Ethical Policy (Best Action per State):")
for i, s in enumerate(states):
    print(f"{s} -> {actions[np.argmax(Q[i])]}")

print("\nEthical Performance Metrics:")
print("Harmful Decisions Made:", harmful_actions)
print("Ethical Compliance Score:", round(1 - harmful_actions / episodes, 2))
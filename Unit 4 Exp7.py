import random
import math
import matplotlib.pyplot as plt

# Environment size
WIDTH, HEIGHT = 100, 100

# Obstacles (rectangular buildings)
obstacles = [
    (20, 20, 20, 40),
    (60, 10, 30, 30),
    (50, 60, 40, 20)
]

START = (10, 10)
GOAL = (90, 90)
STEP_SIZE = 3
MAX_ITER = 3000

tree = {START: None}

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest_node(point):
    return min(tree.keys(), key=lambda n: distance(n, point))

def collision_free(p1, p2):
    for ox, oy, w, h in obstacles:
        if ox <= p2[0] <= ox + w and oy <= p2[1] <= oy + h:
            return False
    return True

def steer(p1, p2):
    theta = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    return (p1[0] + STEP_SIZE * math.cos(theta),
            p1[1] + STEP_SIZE * math.sin(theta))

goal_node = None

for _ in range(MAX_ITER):
    rand_point = (random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
    nearest = nearest_node(rand_point)
    new_node = steer(nearest, rand_point)

    if collision_free(nearest, new_node):
        tree[new_node] = nearest
        if distance(new_node, GOAL) < STEP_SIZE:
            tree[GOAL] = new_node
            goal_node = GOAL
            break

# Extract path
path = []
if goal_node:
    node = goal_node
    while node:
        path.append(node)
        node = tree[node]
    path.reverse()

# Visualization
plt.figure(figsize=(7, 7))

# Draw obstacles
for ox, oy, w, h in obstacles:
    plt.gca().add_patch(plt.Rectangle((ox, oy), w, h, color='gray'))

# Draw tree
for node, parent in tree.items():
    if parent:
        plt.plot([node[0], parent[0]], [node[1], parent[1]], 'g-', linewidth=0.5)

# Draw path
if path:
    px, py = zip(*path)
    plt.plot(px, py, 'r-', linewidth=2, label="UAV Path")

plt.scatter(*START, color='blue', s=100, label="Start")
plt.scatter(*GOAL, color='red', s=100, label="Goal")

plt.title("UAV Surveillance Path Planning using RRT")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.show()
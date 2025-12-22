grid = [
    [0, 0, 1],
    [0, 0, 0],
    [0, 0, 0]
]
actions = [(-1,0),(1,0),(0,-1),(0,1)]  # U, D, L, R
gamma = 0.9
V = [[0]*3 for _ in range(3)]
def reward(x,y):
    return 10 if grid[x][y]==1 else -1
for _ in range(20):
    for i in range(3):
        for j in range(3):
            V[i][j] = max(
                reward(nx,ny) + gamma*V[nx][ny]
                for dx,dy in actions
                if 0 <= (nx:=i+dx) < 3 and 0 <= (ny:=j+dy) < 3
            )
print("State-Value Function:")
for row in V:
    print([round(v,2) for v in row])
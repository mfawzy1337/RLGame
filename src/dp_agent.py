import numpy as np

def value_iteration(env, gamma=0.99, theta=0.0001):
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    grid_size = env.grid_size
    values = np.zeros((grid_size, grid_size))
    policy = np.zeros((grid_size, grid_size), dtype=int)
    
    while True:
        delta = 0
        for x in range(grid_size):
            for y in range(grid_size):
                old_value = values[x, y]
                value_list = []
                for action in actions:
                    env.agent_pos = [x, y]
                    next_state, reward, _ = env.step(action)
                    nx, ny = next_state
                    value_list.append(reward + gamma * values[nx, ny])
                values[x, y] = max(value_list)
                policy[x, y] = np.argmax(value_list)
                delta = max(delta, abs(old_value - values[x, y]))
        
        if delta < theta:
            break
    return policy, values

class GridWorld:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        """Resets the environment and places the agent at start."""
        self.agent_pos = [0, 0]
        self.goal_pos = [self.grid_size - 1, self.grid_size - 1]
        self.obstacles = [[1, 1], [2, 2], [3, 3]]  # Fixed obstacles
        self.reward_positions = [[0, 2], [2, 0], [3, 4]]  # Rewards
        return self.agent_pos

    def step(self, action):
        """Takes an action and returns next_state, reward, done."""
        x, y = self.agent_pos
        if action == "UP":
            x = max(0, x - 1)
        elif action == "DOWN":
            x = min(self.grid_size - 1, x + 1)
        elif action == "LEFT":
            y = max(0, y - 1)
        elif action == "RIGHT":
            y = min(self.grid_size - 1, y + 1)
        
        self.agent_pos = [x, y]
        
        # Check reward
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 10, True  # Reached goal
        elif self.agent_pos in self.obstacles:
            return self.agent_pos, -5, True  # Hit obstacle
        elif self.agent_pos in self.reward_positions:
            return self.agent_pos, 5, False  # Collectible reward
        else:
            return self.agent_pos, -1, False  # Step penalty

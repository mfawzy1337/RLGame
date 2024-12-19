import numpy as np
import matplotlib.pyplot as plt
from game_env import GridWorld
from dp_agent import value_iteration
from td_agent import QLearningAgent
from policy_agent import PolicyGradientAgent
from utils import plot_rewards

def visualize_grid(env):
    """Visualizes the grid with agent, obstacles, rewards, and goal."""
    grid_size = env.grid_size
    agent_pos = env.agent_pos
    goal_pos = env.goal_pos
    obstacles = env.obstacles
    reward_positions = env.reward_positions
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Create the grid with origin (0, 0) at the top-left corner
    ax.set_xticks(np.arange(-0.5, grid_size, 1))
    ax.set_yticks(np.arange(-0.5, grid_size, 1))
    ax.set_xticklabels([])  # Hide x-ticks
    ax.set_yticklabels([])  # Hide y-ticks
    ax.grid(True)

    # Reverse the y-axis to have (0, 0) at the top-left
    ax.invert_yaxis()

    # Plot agent position
    ax.plot(agent_pos[1], agent_pos[0], 'go', markersize=12, label="Agent")
    
    # Plot goal
    ax.plot(goal_pos[1], goal_pos[0], 'rx', markersize=12, label="Goal")
    
    # Plot obstacles
    for obs in obstacles:
        ax.plot(obs[1], obs[0], 'ks', markersize=12, label="Obstacle")
    
    # Plot rewards
    for reward in reward_positions:
        ax.plot(reward[1], reward[0], 'bo', markersize=10, label="Reward")
    
    # Title and legend
    ax.set_title("GridWorld: Agent Position", fontsize=14)
    ax.legend()
    
    plt.show()
    
def manual_play(env):
    """Allows the user to play the GridWorld manually with visualization."""
    print("\nManual Play Mode:")
    print("Controls: W (Up), S (Down), A (Left), D (Right), Q (Quit)\n")
    env.reset()
    done = False

    while not done:
        # Visualize the grid
        visualize_grid(env)
        
        # Ask for action input
        print(f"Agent Position: {env.agent_pos}")
        action = input("Enter action (W/A/S/D): ").upper()

        if action == "W":
            next_state, reward, done = env.step("UP")
        elif action == "S":
            next_state, reward, done = env.step("DOWN")
        elif action == "A":
            next_state, reward, done = env.step("LEFT")
        elif action == "D":
            next_state, reward, done = env.step("RIGHT")
        elif action == "Q":
            print("Exiting Manual Play.")
            break
        else:
            print("Invalid input. Use W/A/S/D to move.")
            continue

        print(f"Next State: {next_state}, Reward: {reward}")
        if done:
            print("Game Over!\n")
            break

def train_and_evaluate(env):
    """Trains and evaluates RL agents."""
    print("\nTraining and Evaluating RL Agents...")

    # Dynamic Programming
    print("\nRunning Dynamic Programming...")
    policy, values = value_iteration(env)
    print("Optimal Values (Dynamic Programming):")
    print(values)

    # Q-Learning
    print("\nTraining Q-Learning Agent...")
    q_agent = QLearningAgent(env)
    q_agent.train()
    print("Q-Learning Training Complete!")
    plot_rewards(q_agent.episode_rewards, "Q-Learning Training Rewards")

    # Policy Gradient
    print("\nTraining Policy Gradient Agent...")
    pg_agent = PolicyGradientAgent(env)
    pg_agent.train()
    print("Policy Gradient Training Complete!")
    plot_rewards(pg_agent.episode_rewards, "Policy Gradient Training Rewards")

def main():
    env = GridWorld(grid_size=4)  # Initialize the environment

    print("Welcome to the GridWorld Game!")
    print("Choose an option:")
    print("1. Play the game manually")
    print("2. Train RL agents and evaluate")

    choice = input("Enter 1 or 2: ")

    if choice == "1":
        manual_play(env)
    elif choice == "2":
        train_and_evaluate(env)
    else:
        print("Invalid choice. Exiting...")

if __name__ == "__main__":
    main()

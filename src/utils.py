import matplotlib.pyplot as plt

def plot_rewards(rewards, title):
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title(title)
    plt.show()

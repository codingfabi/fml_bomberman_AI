import matplotlib.pyplot as plt
import numpy as np


def plotScores(scores):      
    plt.plot(scores)
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.savefig('scores.png')

def plotRewards(rewards):
    plt.plot(rewards)
    plt.ylabel('Rewards')
    plt.xlabel('Game')
    plt.savefig('rewards.png')

scores = np.loadtxt('scores.txt')
plotScores(scores)

rewards = np.loadtxt('rewards.txt')
plotRewards(rewards)


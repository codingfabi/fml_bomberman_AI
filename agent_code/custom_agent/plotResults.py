import matplotlib.pyplot as plt
import numpy as np


def plotScores(scores):      
    plt.plot(scores)
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.show()


scores = np.loadtxt('scores.txt')
print(scores)
plotScores(scores)
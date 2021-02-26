import numpy as np

class CustomModel:
 

    def __init__(self):
        self.actions = np.array(['UP','DOWN','LEFT','RIGHT','BOMB','WAIT'])

    def predict_action(self, game_state):
        weights = np.random.rand(len(self.actions))
        weights = weights / weights.sum()
        print(weights)
        action = np.random.choice(self.actions, p = weights)
        return action
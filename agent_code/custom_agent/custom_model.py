import numpy as np
from .helper import state_to_features

class CustomModel:
 

    def __init__(self):
        self.actions = np.array(['UP','DOWN','LEFT','RIGHT','BOMB','WAIT'])

    def predict_action(self, game_state):
        features = state_to_features(game_state)
        weights = np.random.rand(len(self.actions))
        weights = weights / weights.sum()
        action = np.random.choice(self.actions, p = weights)
        return action
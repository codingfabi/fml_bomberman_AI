import numpy as np
import pickle
import os
import random

from .helper import state_to_features
from .train import setup_training

actions = np.array(['UP','DOWN','LEFT','RIGHT','BOMB','WAIT']) 

def setup(self):
    """
    sets up agent code in training or non-training mode
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up new model")
    else:
        self.logger.info("Loading saved model")
        """ with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file) """



def act(self, game_state: dict):
    
    random_prob = 0.1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        action = np.random.choice(actions, p=[0.2,0.2,0.2,0.2,0.1,0.1])
        self.logger.info('picked this action: ', action)
        return action
    
    self.logger.debug("Querying model for action")
    model_action = self.model.predict_action(game_state)
    self.logger.debug("Model returnd action: ", model_action)
    return model_action
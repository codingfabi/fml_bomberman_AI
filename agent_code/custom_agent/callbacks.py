import numpy as np
import pickle
import os
import random

from .helper import state_to_features
from .train import setup_training, do_training_step

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
    
    if self.train:
        action = do_training_step(self, game_state)
        return action
    
    self.logger.debug("Querying model for action")
    model_action = self.model.predict_action(game_state)
    self.logger.debug("Model returnd action: ", model_action)
    return model_action
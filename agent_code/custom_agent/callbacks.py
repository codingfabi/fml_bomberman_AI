import numpy as np
import pickle
import os
import random

from .helper import state_to_features
from .train import setup_training, do_training_step

from .custom_model import CustomModel

actions = np.array(['UP','DOWN','LEFT','RIGHT','BOMB','WAIT']) 

def setup(self):
    """
    sets up agent code in training or non-training mode
    """
    if self.train:
        self.logger.info("Setting up new model")
    else:
        self.logger.info("Loading saved model")
        self.model = CustomModel()



def act(self, game_state: dict):
    
    if self.train:
        action = do_training_step(self, game_state)
        return action
    
    self.logger.debug("Querying model for action")
    action_index = self.model.predict_action(game_state)
    model_action = self.model.actions[action_index]
    self.logger.debug("Model returnd action: ", model_action)
    return model_action
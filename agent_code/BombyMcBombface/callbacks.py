import os
import pickle
import random

import numpy as np

from .train import do_training_step


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("BombyMcBombface.pt"):
        self.logger.info("Passing to training")
    else:
        self.logger.info("Loading model from saved state.")
        with open("BombyMcBombface.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if self.train:
        action = do_training_step(self, game_state)
        return action
    else:
        self.logger.debug("Querying model for action.")
        predictions = self.model.predict(game_state)
        action = ACTIONS[np.argmax(predictions)]
        return action

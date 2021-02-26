import numpy as np
import pickle
import os

actions = np.array(['UP','DOWN','LEFT','RIGHT','BOMB','WAIT']) 

def setup(self):
    """
    sets up agent code in training or non-training mode
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up new model")
        weights = np.random.rand(len(actions))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading saved model")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)



def act(self, game_state: dict):

    self.model = [0.2,0.2,0.2,0.2,0.1,0.1]

    random_prob = 0.1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        action = np.random.choice(actions, p=[0.2,0.2,0.2,0.2,0.1,0.1])
        self.logger.info('picked this action: ', action)
        return action
    
    self.logger.debug("Querying model for action")
    model_action = np.random.choice(actions, p=self.model)
    self.logger.debug("Model returnd action: ", model_action)
    return model_action

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)

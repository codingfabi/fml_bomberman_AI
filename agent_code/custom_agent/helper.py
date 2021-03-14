import numpy as np
from typing import List

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

    # store features in variable
    round = game_state['round']
    step = game_state['step']
    ownPosition = game_state['self'][3]
    if len(game_state['coins']) > 0:
        firstCoin = game_state['coins'][0]
    else:
        firstCoin = (0,0)



    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(ownPosition)
    channels.append(firstCoin)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return tuple(stacked_channels.reshape(-1))

def action_to_numeric(actions: List[str], action):
    """
    this function maps an array string to a numeric list of actions 
    lets see if we need this
    """
    numericList = np.zeros(len(actions))
    actionIndex = actions.index(action)
    numericList[actionIndex] = 1

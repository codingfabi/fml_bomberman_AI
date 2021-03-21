import numpy as np
from typing import List

import settings as s

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

    channels = []
    """ 
    # store features in variable
    ownPosition = getOwnPosition(game_state)
    channels.append(ownPosition[0])
    channels.append(ownPosition[1])
    channels.append(len(game_state['coins']))
    channels.append(len(game_state['bombs']))
    channels.append(countCrates(game_state['field']))
    # add surrounding fields to channels
    field = game_state['field']
    for coin in game_state['coins']:
        field[coin] = 2
    for bomb in game_state['bombs']:
        bomb_coordinates = bomb[0]
        field[bomb_coordinates] = 3
    surrounding = getSurroundingFields(field, ownPosition)
    for point in surrounding:
        channels.append(point) """

    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    ownPosition = getOwnPosition(game_state)
    
    channels.append(ownPosition[0])
    channels.append(ownPosition[1])
    for row in getCratesMap(field):
        for point in row:
            channels.append(point)
    for row in getCoinsMap(field, coins):
        for point in row:
            channels.append(point)
    for rows in getBombsMap(field, bombs):
        for point in row:
            channels.append(point)

    # For example, you could construct several channels of equal shape, ...
    #channels = []
    #channels.append(ownPosition)
    # channels.append(firstCoin)

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

def getGameNumberFromState(game_state: dict):
    return game_state['round']

def getStepsFromState(game_state:dict):
    return game_state['step']

def getOwnPosition(game_state:dict):
    if game_state is not None:
        return game_state['self'][3]
    else:
        return (0,0)

def getSurroundingFields(field, position):
    myX = position[0]
    myY = position[1]

    surrounding = []
    surrounding.append(field[myX-1,myY-1])
    surrounding.append(field[myX, myY-1])
    surrounding.append(field[myX+1, myY-1])
    surrounding.append(field[myX-1, myY])
    surrounding.append(field[myX+1, myY])
    surrounding.append(field[myX-1, myY+1])
    surrounding.append(field[myX, myY+1])
    surrounding.append(field[myX+1, myY+1])

    return surrounding

def countCrates(field):
    return np.count_nonzero(field == 1)

def getCratesMap(field):
    newField = np.zeros((s.COLS, s.ROWS))
    coordinates = np.where(field == 1)
    for coordinate in coordinates:
        newField[coordinate] = 1
    return newField

def getCoinsMap(field, coins):
    coinField = np.zeros((s.COLS, s.ROWS))
    coordinates = coins
    for coordinate in coordinates:
        coinField[coordinate] = 1
    return coinField

def getBombsMap(field, bombs):
    bombField = np.zeros((s.COLS, s.ROWS))
    coordinates = bombs
    for coordinate in coordinates:
        bombField[coordinate[0]] = 1
    return bombField
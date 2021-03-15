import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import events as e

from .custom_model import CustomModel

from .helper import state_to_features
from .helper import getGameNumberFromState
from .helper import getStepsFromState


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
LEARNING_RATE = 0.001

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

# define the rewards for each game event
game_rewards = {
    e.COIN_COLLECTED: 40,
    e.KILLED_OPPONENT: 100,
    e.MOVED_LEFT: -1,
    e.MOVED_RIGHT: -1,
    e.MOVED_UP: -1,
    e.MOVED_DOWN: -1,
    e.WAITED: -10,
    e.INVALID_ACTION: -50,
    e.BOMB_DROPPED: 1,
    e.BOMB_EXPLODED: 0,
    e.CRATE_DESTROYED: 10,
    e.COIN_FOUND: 20,
    e.KILLED_SELF: -5,
    e.GOT_KILLED: -10,
    e.OPPONENT_ELIMINATED: 5,
    e.SURVIVED_ROUND: 0
    }


def setup_training(self):
    """
    Initialize self for training purpose.
    This is called from 'setup' in callbacks.py
    """

    print('setup training was called')
    self.n_games = 0
    self.epsilon = 1
    self.model = CustomModel()
    self.transitions = []



    # Initilize Q-Table

def do_training_step(self, game_state: dict):
    features = state_to_features(game_state)
    if(random.uniform(0, 1) < self.epsilon):
        action = np.random.choice(self.model.actions)
    else:
        actionIndex = self.model.predict_action(game_state)
        action = self.model.actions[actionIndex]

    #self.logger.debug(f'Action taken:', action)

    reduce_epsilon(self, getGameNumberFromState(game_state),getStepsFromState(game_state))

    return action
    

    
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    :param self: standard object that is passed to all methods
    :param old_game_state: The state that was passed to the last call of `act`
    :param self_action: The action taken by the agent
    :param new_game_state: The state the agent is in now
    :param events: Diff between old and new game_state
    """

    """    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    self.logger.debug(f'The total reward for the events was', reward_from_events(self, events)) """

    """ current_transition = Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)) """
    # here we could add some events to add rewards
    rewardsum = reward_from_events(self, events)

    self.model.update_qtable(old_game_state,new_game_state,self_action,rewardsum)

    """ self.transitions.append(current_transition)
    f = open("demo.txt", "a")
    if current_transition.action != None:
        f.write(current_transition.action)
    f.close() """


def end_of_round(self, laste_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards
    """
    
    """self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')"""

    rewards = reward_from_events(self, events)

    self.model.update_qtable_after_game_ends(laste_game_state, last_action, rewards)

    print('Your score for the game was: ')
    print(laste_game_state['self'][1])


def reward_from_events(self, events: List[str]) -> int:
    """
    this defines the rewards set for each event defined in the world
    all possible events are:
        e.MOVED_LEFT
        e.MOVED_RIGHT
        e.MOVED_UP
        e.MOVED_DOWN
        e.WAITED
        e.INVALID_ACTION
        e.BOMB_DROPPED
        e.BOMB_EXPLODED
        e.CRATE_DESTROYED
        e.COIN_FOUND
        e.COIN_COLLECTED
        e.KILLED_OPPONENT
        e.KILLED_SELF
        e.GOT_KILLED
        e.OPPONENT_ELIMINATED
        e.SURVIVED_ROUND
    """
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def reduce_epsilon(self, gamesPlayed: int, steps: int):
    
    if gamesPlayed > 50 and steps < 10:
        self.epsilon = 0.02
    else:
        if gamesPlayed > 25:
            self.epsilon = 0.8
        if gamesPlayed > 50:
            self.epsion = 0.7
        if gamesPlayed > 100:
            self.epsilon = 0.6    
        if gamesPlayed > 150:
            self.epsilon = 0.5
        if gamesPlayed > 200:
            self.epsilon = 0.1
        if gamesPlayed > 250:
            self.epsilon = 0.2
        if gamesPlayed > 350:
            self.epsion = 0.1
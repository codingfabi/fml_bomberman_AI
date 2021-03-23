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
from .helper import getOwnPosition


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
LEARNING_RATE = 0.001

# Events
VISITED_SAME_PLACE = "VISITED_SAME_PLACE"

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
    e.SURVIVED_ROUND: 0,
    VISITED_SAME_PLACE: -15
}

gameResults = []


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
    self.gameResults = []
    self.episodeReward = 0



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

    newPosition = getOwnPosition(new_game_state)

    total_events = addAuxiliaryEvents(self, newPosition, events)

    # here we could add some events to add rewards
    rewardsum = reward_from_events(self, total_events)

    self.episodeReward += rewardsum

    self.model.update_qtable(old_game_state,new_game_state,self_action,rewardsum)

    # update models last position AFTER handing out auxilliary rewards
    self.model.updateLastPositions(newPosition)

    """ self.transitions.append(current_transition)
    f = open("demo.txt", "a")
    if current_transition.action != None:
        f.write(current_transition.action)
    f.close() """


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards
    """
    
    """self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')"""

    rewards = reward_from_events(self, events)

    self.episodeReward += rewards

    self.model.update_qtable_after_game_ends(last_game_state, last_action, rewards)

    score = last_game_state['self'][1]
    
    with open("scores.txt", "a") as scores_log:
        scores_log.write(str(score) + "\t")

    with open('rewards.txt', 'a') as reward_log:
        reward_log.write(str(self.episodeReward) + "\t")


    print('Your score for the game was: ')
    print(score)

    # reset episode reward
    self.episodeReward = 0


def reward_from_events(self, events: List[str]) -> int:
    reward_sum = 0
    for event in events:
        if event == VISITED_SAME_PLACE:
            print('reward for loop was handed out')
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def reduce_epsilon(self, gamesPlayed: int, steps: int):
    
    if gamesPlayed < 1000 and  gamesPlayed > 300 and steps < 10:
        self.epsilon = 0.02
    else:
        if gamesPlayed > 200:
            self.epsilon = 0.8
        if gamesPlayed > 200:
            self.epsion = 0.7
        if gamesPlayed > 200:
            self.epsilon = 0.6    
        if gamesPlayed > 250:
            self.epsilon = 0.5
        if gamesPlayed > 400:
            self.epsilon = 0.3
        if gamesPlayed > 750:
            self.epsilon = 0.2
        if gamesPlayed > 1000:
            self.epsilon = 0

def addAuxiliaryEvents(self, newPosition: tuple, events: List[str]):
    """
    The idea of this function is to add some auxiliary rewards in order to improve the agents behavior
    """
    newEvents = events
    # Idea: add a penalty if the same place has been visited in the last two states. This should help to prevent loops
    position = newPosition
    if position in self.model.lastPositions:
        newEvents.append(VISITED_SAME_PLACE)

    return newEvents
    

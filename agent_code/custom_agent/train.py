import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import events as e

from .custom_model import CustomModel
from .cstom_model import update_qtable

from .helper import state_to_features

from .callbacks import actions

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
    e.COIN_COLLECTED: 10,
    e.KILLED_OPPONENT: 50,
    e.MOVED_LEFT: -1,
    e.MOVED_RIGHT: -1,
    e.MOVED_UP: -1,
    e.MOVED_DOWN: -1,
    e.WAITED: -0.1,
    e.INVALID_ACTION: -10,
    e.BOMB_DROPPED: 0.1,
    e.BOMB_EXPLODED: 0,
    e.CRATE_DESTROYED: 1,
    e.COIN_FOUND: 1,
    e.KILLED_SELF: -10,
    e.GOT_KILLED: -50,
    e.OPPONENT_ELIMINATED: 5,
    e.SURVIVED_ROUND: 5
    }


def setup_training(self):
    """
    Initialize self for training purpose.
    This is called from 'setup' in callbacks.py
    """
    self.n_games = 0
    self.epsilon = 0.3
    self.gamma = 0.8
    self.alpha = 0.1
    self.model = CustomModel()
    self.transitions = []



    # Initilize Q-Table

def do_training_step(self, game_state: dict):
    features = state_to_features(game_state)
    if(random.uniform(0, 1) < self.epsilon):
        action = np.random.choice(actions)
    else:
        self.model.predict_action(self, game_state)
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

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    print('events occured was called')
    print(reward_from_events(self, events))

    """ current_transition = Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)) """
    # here we could add some events to add rewards
    rewardsum = reward_from_events(events)
    self.model.update_qtable(self, action_taken=self_action, old_state = old_game_state, next_state = new_game_state, total_reward  = reward_sum)

    """ self.transitions.append(current_transition)
    f = open("demo.txt", "a")
    if current_transition.action != None:
        f.write(current_transition.action)
    f.close() """


def end_of_round(self, laste_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards
    """

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    self.transitions.append(Transition(state_to_features(laste_game_state), last_action, None, reward_from_events(self, events)))

    # store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


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
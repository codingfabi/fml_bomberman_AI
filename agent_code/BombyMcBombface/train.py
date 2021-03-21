import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List
import os

import events as e

from .model import ValueFunctionApproximator

from .helper import state_to_features
from .helper import getGameNumberFromState
from .helper import getStepsFromState
from .helper import getOwnPosition

from sklearn.utils.validation import check_is_fitted
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 400  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
LEARNING_RATE = 0.001
BATCH_SIZE = 50

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_SPACE = len(ACTIONS)

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
    e.INVALID_ACTION: -100,
    e.BOMB_DROPPED: 1,
    e.BOMB_EXPLODED: 0,
    e.CRATE_DESTROYED: 10,
    e.COIN_FOUND: 20,
    e.KILLED_SELF: -5,
    e.GOT_KILLED: -10,
    e.OPPONENT_ELIMINATED: 5,
    e.SURVIVED_ROUND: 0,
    VISITED_SAME_PLACE: -5
}


def setup_training(self):
    """
    Initialize self for training purpose.
    This is called from 'setup' in callbacks.py
    """

    # TODO setup training
    self.lastPositions = [(np.NINF, np.NINF), (np.inf,np.inf)]
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.gamma = 0.8
    self.epsilon = 1

    print('setup training was called')

    self.lastBatch = []

    # check for setting up of new model or retraining existing model
    if not os.path.isfile('BombyMcBombface.pt'):
        self.approximator = ValueFunctionApproximator(ACTION_SPACE)
    else:
        with open("BombyMcBombface.pt", "rb") as file:
            self.approximator = pickle.load(file)

def do_training_step(self, game_state: dict):
    
    #policy_function = create_policy_for_approximator(self, self.approximator, self.epsilon, ACTION_SPACE)

    #policy = policy_function(game_state)
    if(random.uniform(0, 1) < self.epsilon):
        policy = [0.2,0.2,0.2,0.2,0.1,0.1]
        action = np.random.choice(ACTIONS, p = policy)
    else:
        values = self.approximator.predict(game_state)
        action = ACTIONS[np.argmax(values)]

    reduce_epsilon(self, getGameNumberFromState(game_state), getStepsFromState(game_state))

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
    updateLastPositions(self, newPosition)

    total_events = addAuxiliaryEvents(self, newPosition, events, old_game_state, new_game_state)

    # here we could add some events to add rewards
    current_transition = Transition(old_game_state, self_action, new_game_state, total_events)
    self.transitions.append(current_transition)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards
    """
    
    """self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')"""

    if len(self.transitions) < BATCH_SIZE:
        return

    rewards = reward_from_events(self, events)

    episode_reward = rewards
    batch = random.sample(self.transitions, BATCH_SIZE)
    if batch == self.lastBatch:
        print('batch was the same')
    
    self.lastBatch = batch

    counter = 0

    #policy = create_policy_for_approximator(self.approximator, self.epsilon, ACTION_SPACE )
    
    # update the model (episodic updating)
    for game_step in batch:
        counter += 1
        state = game_step[0]
        action = game_step[1]
        next_state = game_step[2]
        episode_events = game_step[3]

        episode_reward += reward_from_events(self, episode_events)

        # do this because first state is somehow empty
        if state is not None and action is not None:

            actionIndex = ACTIONS.index(action)
            
            models_are_fitted = True
            
            # check if models are fitted
            for model in self.approximator.models:
                if hasattr(model, 'coef_'):
                    pass
                else:
                    print('a model was not fitted')
                    models_are_fitted = False

            if models_are_fitted:
                next_q_values = self.approximator.predict(next_state)
                temporalDifference_target = episode_reward + self.gamma * np.max(next_q_values)
            else:
                temporalDifference_target = 0

            #print(temporalDifference_target)

            # use target to update approximator
            self.approximator.updateEstimatorParameters(state, actionIndex, temporalDifference_target)

    print('transitions used for training: ', counter)
    
    with open("rewards.txt", "a") as rewards_log:
            rewards_log.write(str(rewards) + "\t")
    
    score = last_game_state['self'][1]
    
    with open("scores.txt", "a") as scores_log:
        scores_log.write(str(score) + "\t")


    print('Your score for the game was: ')
    print(score)

    self.logger.info("Dumping Model")
    with open("BombyMcBombface.pt", "wb") as file:
        pickle.dump(self.approximator, file)

    


def reward_from_events(self, events: List[str]) -> int:
    reward_sum = 0
    for event in events:
        #if event == VISITED_SAME_PLACE:
            #print('reward for loop was handed out')
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def reduce_epsilon(self, gamesPlayed: int, steps: int):
    
    oldEpsilon = self.epsilon

    if gamesPlayed > 25 and steps < 20:
        self.epsilon = 0.1
    if gamesPlayed > 25 and steps < 50:
        self.epsilon = 0.5
    if gamesPlayed > 25 and steps < 100:
        self.epsilon = 0.4
    if gamesPlayed > 50 and steps < 400:
        self.epsilon = 0.1
    if gamesPlayed >300:
        self.epsilon = 0.01

    else:
        if gamesPlayed > 1000:
            self.epsilon = 0.1
        if gamesPlayed > 2000:
            self.epsilon = 0.6    
        if gamesPlayed > 3500:
            self.epsilon = 0.5
        if gamesPlayed > 5000:
            self.epsilon = 0.3
        if gamesPlayed > 7500:
            self.epsilon = 0.1
        if gamesPlayed > 1000:
            self.epsilon = 0.1

    if oldEpsilon is not self.epsilon:
        print('epsilon reduced')

def addAuxiliaryEvents(self, newPosition: tuple, events: List[str], old_game_state: dict, new_game_state: dict):
    """
    The idea of this function is to add some auxiliary rewards in order to improve the agents behavior
    """
    newEvents = events
    # Idea: add a penalty if the same place has been visited in the last two states. This should help to prevent loops
    position = newPosition
    if position in self.lastPositions:
        newEvents.append(VISITED_SAME_PLACE)

    return newEvents
    
def create_policy_for_approximator(self, approximator: ValueFunctionApproximator, epsilon, actionSpace):
    """
    returns a policy function
    """
    def policy(observation):
        A = np.ones(actionSpace, dtype=float) * epsilon / actionSpace
        q_values = approximator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy 

def updateLastPositions(self, position: tuple):
    self.lastPositions[0] = self.lastPositions[1]
    self.lastPositions[1] = position
    
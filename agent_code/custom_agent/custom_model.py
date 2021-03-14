import numpy as np
from .helper import state_to_features
import time
import os

class CustomModel:

    qTableFileName: str 
    stateTableFileName: str
    gamma: int
    alpha: int 

    def __init__(self):
        print('new modell created')
        self.actions = np.array(['UP','DOWN','LEFT','RIGHT','BOMB','WAIT'])
        self.gamma = 0.8
        self.alpha = 0.1

        
        self.stateTableFileName = "custom_model_states.npy"
        self.qTableFileName = "custom_model_qvalues.npy"

        if not os.path.isfile(self.stateTableFileName):
            print('create new states table')
            np.save(self.stateTableFileName, np.empty((1,4)))

        
        if not os.path.isfile(self.qTableFileName):
            print('create new qtable')
            np.save(self.qTableFileName, np.empty((1,len(self.actions))))


    def predict_action(self, game_state):
        features = state_to_features(game_state)
        q_table = np.load("custom_model.npy")
        state_index = np.where(q_table)
        action = np.argmax(q_table[:,0] == features)
        """ weights = np.random.rand(len(self.actions))
        weights = weights / weights.sum()
        action = np.random.choice(self.actions, p = weights) """
        return action

    
    def stateExistsInTable(self, state, table):
        return next((True for elem in table if np.array_equal(elem, state)), False)

    
    def getIndexOfStateInStatesTable(self, statesTable, features):
        return np.where(statesTable == features)[0]
    
    
    def update_qtable(self, old_state, next_state, action_taken, total_reward):
        newTable = np.load(self.qTableFileName)
        statesTable = np.load(self.stateTableFileName)
        actionIndex = np.where(self.actions == action_taken)[0]        
        
        old_state_features = state_to_features(old_state)
        next_state_features = state_to_features(next_state)

        if type(old_state_features) is not np.ndarray:
            print('quitted q table update because old state was shit')
            return None
        
        if type(next_state_features) is not np.ndarray:
            print('quitted q table update because next state was shit')
            return None

        check = next((True for elem in statesTable if np.array_equal(elem, old_state_features)), False)
        if self.stateExistsInTable(old_state_features, statesTable):
            print('encountered old state: ', old_state_features)
            print(old_state_features in statesTable)
            print(statesTable)
            oldStateIndex = self.getIndexOfStateInStatesTable(statesTable, old_state_features)
            old_reward = newTable[oldStateIndex[0], actionIndex]

            if next_state_features in statesTable:
                nextStateIndex = self.getIndexOfStateInStatesTable(statesTable, next_state_features)[0]
                max_value_of_next_state = np.max(newTable[nextStateIndex])
            else:
                max_value_of_next_state = 0

            # this is the essential q_learning function
            new_value = (1 - self.alpha)* old_reward + self.alpha * (total_reward + self.gamma * max_value_of_next_state)
            newTable[oldStateIndex, actionIndex] = new_value

        else:
            newRow = np.zeros([len(self.actions)])
            newRow[actionIndex] = total_reward
            print('should have appended column')
            newTable = np.vstack((newTable, newRow))
            print(newTable)
            statesTable = np.vstack((statesTable, old_state_features))
        
        np.save(self.qTableFileName, newTable)
        np.save(self.stateTableFileName, statesTable)
        
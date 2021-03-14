import numpy as np
from .helper import state_to_features
import time
import os
import pickle

class CustomModel:

    modelDict = {(0,0,0,0):[0,0,0,0,0,0]}
    qDictFileName = str
    qTableFileName: str 
    stateTableFileName: str
    gamma: int
    alpha: int 

    def __init__(self):
        print('new modell created')
        self.actions = np.array(['UP','DOWN','LEFT','RIGHT','BOMB','WAIT'])
        self.gamma = 0.8
        self.alpha = 0.1

        self.qDictFileName = "custom_model.pt"
        self.stateTableFileName = "custom_model_states.npy"
        self.qTableFileName = "custom_model_qvalues.npy"

        if not os.path.isfile(self.stateTableFileName):
            print('create new states table')
            np.save(self.stateTableFileName, np.empty((1,4)))

        
        if not os.path.isfile(self.qTableFileName):
            print('create new qtable')
            np.save(self.qTableFileName, np.empty((1,len(self.actions))))

        if not os.path.isfile(self.qDictFileName):
            print('created new dict file')
            pickle.dump(self.modelDict, open(self.qDictFileName, "wb"))


    def predict_action(self, game_state):
        features = state_to_features(game_state)
        qTable = pickle.load(open(self.qDictFileName, "rb"))
        
        rewards = qTable[state_to_features(game_state)]
        action = np.argmax(rewards)
        """ weights = np.random.rand(len(self.actions))
        weights = weights / weights.sum()
        action = np.random.choice(self.actions, p = weights) """
        return action

    
    def stateExistsInTable(self, state, table):
        return next((True for elem in table if np.array_equal(elem, state)), False)

    
    def getIndexOfStateInStatesTable(self, statesTable, features):
        return np.where(statesTable == features)[0]
    
    
    def update_qtable(self, old_state, next_state, action_taken, total_reward):

        currentQTable = pickle.load(open(self.qDictFileName, "rb" ))

        """ newTable = np.load(self.qTableFileName)
        statesTable = np.load(self.stateTableFileName) """
        actionIndex = np.where(self.actions == action_taken)[0]        
        
        old_state_features = state_to_features(old_state)
        next_state_features = state_to_features(next_state)

        if type(old_state_features) is not tuple:
            print('quitted q table update because old state was shit')
            return None
        
        if type(next_state_features) is not tuple:
            print('quitted q table update because next state was shit')
            return None

        if old_state_features in currentQTable:
            old_state_rewards = currentQTable[old_state_features]
            old_reward = old_state_rewards[actionIndex]

            if next_state_features in currentQTable:
                next_state_rewards = currentQTable[next_state_features]
                max_value_of_next_state = np.max(next_state_rewards)
            else:
                max_value_of_next_state = 0

            # this is the essential q_learning function
            new_value = (1 - self.alpha)* old_reward + self.alpha * (total_reward + self.gamma * max_value_of_next_state)
            currentQTable[old_state_features][actionIndex] = new_value

        else:
            newRow = np.zeros([len(self.actions)])
            newRow[actionIndex] = total_reward
            currentQTable.update({old_state_features: newRow})
        
        pickle.dump(currentQTable, open(self.qDictFileName,"wb"))
        
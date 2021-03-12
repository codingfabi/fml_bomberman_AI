import numpy as np
from .helper import state_to_features

class CustomModel:
 

    def __init__(self):
        self.actions = np.array(['UP','DOWN','LEFT','RIGHT','BOMB','WAIT'])
        self.gamma = 0.8
        self.alpha = 0.1
        np.save("custom_model.npy", np.empty((1,len(self.actions)+1)))
        test = np.load("custom_model.npy")
        print(test)

    def predict_action(self, game_state):
        features = state_to_features(game_state)
        q_table = np.load("custom_model.npy")
        state_index = np.where(q_table)
        action = np.argmax(q_table[:,0] == features)
        """ weights = np.random.rand(len(self.actions))
        weights = weights / weights.sum()
        action = np.random.choice(self.actions, p = weights) """
        return action
    
    def update_qtable(self, old_state, next_state, action_taken, total_reward):
        newTable = np.load("custom_model.npy")
        actionIndex = np.where(self.actions == action_taken)
        old_state_features = state_to_features(old_state)
        next_state_features = state_to_features(next_state)

        oldRewardIndex = self.getIndexOfStateInQTable(newTable, old_state_features)

        if oldRewardIndex[0] > -1:
            old_reward = q_table[oldRewardIndex, actionIndex]

            if next_state_features in newTable:
                max_value_of_next_state = np.max(newTable[next_state_features, :])
            else:
                max_value_of_next_state = 0

            # this is the essential q_learning function
            new_value = (1 - self.alpha)* old_reward + self.alpha * (total_reward + self.gamma * max_value_of_next_state)
            newTable[old_state_features, actionIndex] = new_value

            np.save("custom_model.npy", newTable)
        else:
            newRowValues = np.zeros([len(self.actions)])
            newRowValues[actionIndex] = total_reward
            newTable = np.append(newTable, [old_state, newRowValues])
        

    def getIndexOfStateInQTable(self, qtable, features):
        return np.where(qtable[:, 0] == features)
    
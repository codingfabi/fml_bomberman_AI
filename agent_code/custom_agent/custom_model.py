import numpy as np
from .helper import state_to_features

class CustomModel:
 

    def __init__(self):
        self.actions = np.array(['UP','DOWN','LEFT','RIGHT','BOMB','WAIT'])
        self.gamma = 0.8
        self.alpha = 0.1
        np.save("custom_model.npy", np.array([1,len(self.actions)])) 

    def predict_action(self, game_state):
        features = state_to_features(game_state)
        qtable = np.load("custom_model.npy")
        action = np.argmax(q_table[state])
        """ weights = np.random.rand(len(self.actions))
        weights = weights / weights.sum()
        action = np.random.choice(self.actions, p = weights) """
        return action
    
    def update_qtable(self, old_state, next_state, action_taken, total_reward):
        newTable = np.load("custom_model.npy")
        actionIndex = self.actions.index(action_taken)
        old_state = state_to_features(old_state)
        next_state = state_to_features(new_state)

        if old_state in newTable:
            old_reward = q_table[old_state, actionIndex]

            if new_state in newTable:
                max_value_of_next_state = np.max(newTable[next_state])
            else:
                max_value_of_next_state = 0

            # this is the essential q_learning function
            new_value = (1 - self.alpha)* old_reward + self.alpha * (total_reward + self.gamma * max_value_of_next_state)
            newTable[old_state, actionIndex] = new_value

            np.save("custom_model.npy", newTable)
        else:
            newRowValues = np.array([actions.len])
            newRowValues[actionIndex]=rewards
            newTable = np.append(newTable, [old_state, newRowValues])
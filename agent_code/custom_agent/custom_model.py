import numpy as np
from .helper import state_to_features

class CustomModel:
 

    def __init__(self):
        self.actions = np.array(['UP','DOWN','LEFT','RIGHT','BOMB','WAIT'])
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
        if old_state in newTable:
            old_reward = q_table[old_state, action_taken]
            max_value_of_next_state = np.max(newTable[next_state])

            new_value = (1 - self.alpha)* old_reward + self.alpha * (total_reward + self.gamma * max_value_of_next_state)
            newTable[old_state, action_taken] = new_value

            np.save("custom_model.npy", newTable)
        else:
            newRowValues = np.array([actions.len])
            newRowValues[action_taken]=rewards
            newTable = np.append(newTable, [old_state, newRowValues])
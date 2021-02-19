import numpy as np

actions = np.array(['UP','DOWN','LEFT','RIGHT','BOMB','WAIT']) 

def setup(self):
    pass



def act(self, game_state: dict):
    action = np.random.choice(actions)
    self.logger.info('picked this action: ', action)
    return action
import numpy as np

"""
this is just a file to check on the qTable
"""

qtable = np.load('custom_model_qvalues.npy', allow_pickle=True)
print(qtable)

statesTable = np.load('custom_model_states.npy', allow_pickle=True)
print(statesTable)
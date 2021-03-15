import numpy as np
import pickle
"""
this is just a file to check on the qTable
"""

qtable = pickle.load(open("custom_model.pt","rb"))
print(len(qtable))

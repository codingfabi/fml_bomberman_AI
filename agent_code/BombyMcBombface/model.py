import pickle
import os

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import normalize

from .helper import state_to_features
from .initialMockState import initialMockState

class ValueFunctionApproximator():
    """
    Implements Q-Learning with Value Approximation using stochastik
    gradient decent as a regressor
    """
    def __init__(self, actionSpace):
        """
        split the models according to actions (somthing like a regression tree)
        later execute the action with the biggest prediction
        """
        self.models = []
        for _ in range(actionSpace):
            model = SGDRegressor(loss='huber', alpha=0.00000001, max_iter=10000000)
            # initial modal fit, this is needed to prevent crash in train
            self.models.append(model)
        
        pickle.dump( self, open( "BombyMcBombface.pt", "wb" ) )

    def predict(self, game_state, action = None):
        """
        predicts an action according to the current value function
        """
        features = [state_to_features(game_state)]

        if action == None:
            predictions = []
            for model in self.models:
                prediction = model.predict(features)[0]
                predictions.append(prediction)
            return predictions
        else:
            prediction = self.models[action].predict([features])[0]
    
    def updateEstimatorParameters(self, game_state, action, target):
        """
        update parameters of given state and action
        """
        features = [state_to_features(game_state)]
        self.models[action].fit(features, [target])


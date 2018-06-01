import sys
sys.path.insert(0, '../decision_tree/')

import pandas as pd
import numpy as np

from decision_tree_regressor import DecisionTreeRegressor


class RandomForestRegressor():
    def __init__(self, x, y, n_estimators=10, n_sample=None):
        self.n_estimators = n_estimators
        self.x = x
        self.y = y
        self.n_sample = n_sample if n_sample else len(self.y)
        self.trees = list()
        self.trees = [self.build_tree() for _ in range(self.n_estimators)]
        
    def build_tree(self):
        idxs = np.random.permutation(len(self.y))[:self.n_sample]
        return DecisionTreeRegressor(self.x, self.y, idxs)
    
    def predict(self, x):
        preds = np.array([t.predict(x) for t in self.trees])
        return np.mean(preds, axis=0)
    
    def __repr__(self):
        return "Random Forest Regressor with {} trees".format(self.n_estimators)
        
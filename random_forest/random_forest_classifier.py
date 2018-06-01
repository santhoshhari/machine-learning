import sys
sys.path.insert(0, '../decision_tree/')

import pandas as pd
import numpy as np
from scipy.stats import mode

from decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier():
    def __init__(self, x, y, n_estimators=10, n_sample=None):
        self.n_estimators = n_estimators
        self.x = x
        self.y = y
        self.n_sample = n_sample if n_sample else len(self.y)
        self.trees = list()
        self.trees = [self.build_tree() for _ in range(self.n_estimators)]
        
    def build_tree(self):
        idxs = np.random.permutation(len(self.y))[:self.n_sample]
        return DecisionTreeClassifier(self.x, self.y, idxs)
    
    def predict(self, x):
        preds = np.array([t.predict(x) for t in self.trees])
        return [mode(preds[:,i])[0][0] for i in range(preds.shape[1])]
    
    def __repr__(self):
        return "Random Forest with {} trees".format(self.n_estimators)
        
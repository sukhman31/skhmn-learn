import numpy as np
import math
from enum import Enum

class Impurity(Enum):
    GINI = 1
    ENTROPY = 2

class DecisionTrees:
    
    def __init__(self):
        None

    def gini_impurity(self, labels: np.ndarray):
        impurity = 1.0
        unique_labels = {}
        for label in labels:
            unique_labels[label] = unique_labels.get(label,0) + 1
        for label in unique_labels.keys():
            impurity -= (unique_labels[label]/len(labels))**2
        return impurity
    
    def entropy(self, labels: np.ndarray):
        entropy = 0.0
        unique_labels = {}
        for label in labels:
            unique_labels[label] = unique_labels.get(label,0) + 1
        for label in unique_labels.keys():
            frac = unique_labels[label]/len(labels)
            entropy -= frac*math.log2(frac)
        return entropy
    
    def weighted_impurity(self, groups: list[list[int]], type: Impurity):
        wi = 0
        total = sum(len(group) for group in groups)
        if type == Impurity.GINI:
            for group in groups:
                wi += len(group)*self.gini_impurity(group)/total
        elif type == Impurity.ENTROPY:
            for group in groups:
                wi += len(group)*self.entropy(group)/total
        return wi
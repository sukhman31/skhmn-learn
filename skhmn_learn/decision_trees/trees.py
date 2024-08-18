import numpy as np
import math
from enum import Enum

class Impurity(Enum):
    GINI = 1
    ENTROPY = 2

class Node:

    def __init__(self):
        self.is_leaf_node = False
        self.left_child = None
        self.right_child = None

class DecisionTrees:
    
    def __init__(self):
        None

    def train_tree(self, training_data: np.ndarray, training_class: np.ndarray):
        parent = self.recursive_build_tree(training_data, training_class, Node())
    
    def recurive_build_tree(self, training_data: np.ndarray, training_class: np.ndarray, root: Node):
        best_split = self.get_best_split()
    
    def get_best_split(self, training_data: np.ndarray, training_class: np.ndarray):
        for index in range(training_data.shape[0]):
            split = self.get_split(training_data, training_class, index)
    
    def get_split(self, training_data: np.ndarray, training_class: np.ndarray, index: int):
        

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
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
        self.training_data = None
        self.trainig_class = None

    def __init__(self, is_leaf_node, left_child, right_child, training_data, training_class):
        self.is_leaf_node = is_leaf_node
        self.left_child = left_child
        self.right_child = right_child
        self.training_data = training_data
        self.trainig_class = training_class

class DecisionTrees:
    
    def __init__(self):
        self.parent = None

    def train_tree(self, training_data: np.ndarray, training_class: np.ndarray):
        self.parent = self.recurive_build_tree(training_data, training_class)
    
    def recurive_build_tree(self, training_data: np.ndarray, training_class: np.ndarray):
        if len(np.unique(training_class)) == 1:
            return Node(True, None, None, training_data, training_class)
        left_split, right_split = self.get_best_split(training_data, training_class)
        return Node(False, self.recurive_build_tree(left_split[0], left_split[1]), self.recurive_build_tree(right_split[0], right_split[1]), training_data, training_class)
    
    def get_best_split(self, training_data: np.ndarray, training_class: np.ndarray):
        min_impurity = float('inf')
        best_index = None
        best_entry = None
        for index in range(training_data.shape[1]):
            unique_entries = np.sort(np.unique(training_data[:, index]))
            for entry in unique_entries:
                left_split, right_split = self.get_split(training_data, training_class, index, entry)
                imp = self.weighted_impurity([left_split[1], right_split[1]], Impurity.GINI)
                if imp < min_impurity:
                    min_impurity = imp
                    best_index = index
                    best_entry = entry
        return self.get_split(training_data, training_class, best_index, best_entry)
    
    def get_split(self, training_data: np.ndarray, training_class: np.ndarray, index: int, entry):
        data_index = training_data[:, index]
        if training_data[:, index].dtype.kind in ['i', 'f']:
            mask = data_index >= entry
        else:
            mask = data_index == entry
        return [training_data[~mask, :], training_class[~mask]], [training_data[mask, :], training_class[mask]]


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
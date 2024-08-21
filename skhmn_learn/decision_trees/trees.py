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
        self.index = None
        self.value = None
        self.classification = None

    def __init__(self, is_leaf_node, left_child, right_child, training_data, training_class, index, value, classification):
        self.is_leaf_node = is_leaf_node
        self.left_child = left_child
        self.right_child = right_child
        self.training_data = training_data
        self.trainig_class = training_class
        self.index = index
        self.value = value
        self.classification = classification

class DecisionTrees:
    
    def __init__(self):
        self.parent = None

    def train_tree(self, training_data: np.ndarray, training_class: np.ndarray):
        self.parent = self.recurive_build_tree(training_data, training_class)
    
    def train_regression_tree(self, training_data: np.ndarray, training_val: np.ndarray):
        self.parent = self.recursive_build_regression_tree(training_data, training_val)
    
    def classify(self, eval_data: np.ndarray):
        return self.recursive_walk(eval_data, self.parent)
    
    def evaluate(self, eval_data: np.ndarray):
        return self.recursive_regression_walk(eval_data, self.parent)
    
    def recursive_regression_walk(self, eval_data: np.ndarray, root: Node):
        if root.is_leaf_node:
            return np.average(root.trainig_class)
        if eval_data[root.index].dtype.kind in ['i', 'f']:
            if eval_data[root.index] >= root.value:
                return self.recursive_walk(eval_data, root.right_child)
            else:
                return self.recursive_walk(eval_data, root.left_child)
        else:
            if eval_data[root.index] == root.value:
                return self.recursive_walk(eval_data, root.right_child)
            else:
                return self.recursive_walk(eval_data, root.left_child)
    
    def recursive_walk(self, eval_data: np.ndarray, root: Node):
        if root.is_leaf_node:
            return root.classification
        if eval_data[root.index].dtype.kind in ['i', 'f']:
            if eval_data[root.index] >= root.value:
                return self.recursive_walk(eval_data, root.right_child)
            else:
                return self.recursive_walk(eval_data, root.left_child)
        else:
            if eval_data[root.index] == root.value:
                return self.recursive_walk(eval_data, root.right_child)
            else:
                return self.recursive_walk(eval_data, root.left_child)
    
    def recurive_build_tree(self, training_data: np.ndarray, training_class: np.ndarray):
        if len(np.unique(training_class)) == 1:
            return Node(True, None, None, training_data, training_class, None, None, np.unique(training_class)[0])
        left_split, right_split, index, value = self.get_best_split(training_data, training_class)
        return Node(False, self.recurive_build_tree(left_split[0], left_split[1]), self.recurive_build_tree(right_split[0], right_split[1]), training_data, training_class, index, value, None)
    
    def recursive_build_regression_tree(self, training_data: np.ndarray, training_val: np.ndarray):
        if len(training_val) == 2:
            return Node(True, None, None, training_data, training_val, None, None, None)
        left_split, right_split, index, value = self.get_best_split(training_data, training_val)
        return Node(False, self.recurive_build_tree(left_split[0], left_split[1]), self.recurive_build_tree(right_split[0], right_split[1]), training_data, training_val, index, value, None)
    
    def get_best_split(self, training_data: np.ndarray, training_class: np.ndarray):
        min_impurity = float('inf')
        best_index = None
        best_entry = None
        for index in range(training_data.shape[1]):
            unique_entries = np.sort(np.unique(training_data[:, index]))
            for entry in unique_entries:
                left_split, right_split, _, _ = self.get_split(training_data, training_class, index, entry)
                imp = self.weighted_impurity([left_split[1], right_split[1]], Impurity.GINI)
                if imp < min_impurity:
                    min_impurity = imp
                    best_index = index
                    best_entry = entry
        return self.get_split(training_data, training_class, best_index, best_entry)
    
    def get_best_regression_split(self, training_data: np.ndarray, training_val: np.ndarray):
        min_variance = float('inf')
        best_index = None
        best_entry = None
        for index in range(training_data.shape[1]):
            unique_entries = np.sort(np.unique(training_data[:, index]))
            for entry in unique_entries:
                left_split, right_split, _, _ = self.get_split(training_data, training_val, index, entry)
                var = self.weighted_variance([left_split[1], right_split[1]])
                if var < min_variance:
                    min_variance = var
                    best_index = index
                    best_entry = entry
        return self.get_split(training_data, training_val, best_index, best_entry)
    
    def get_split(self, training_data: np.ndarray, training_class: np.ndarray, index: int, entry):
        data_index = training_data[:, index]
        if training_data[:, index].dtype.kind in ['i', 'f']:
            mask = data_index >= entry
        else:
            mask = data_index == entry
        return [training_data[~mask, :], training_class[~mask]], [training_data[mask, :], training_class[mask]], index, entry


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
    
    def variance(self, labels: np.ndarray):
        return np.var(labels)
    
    def weighted_variance(self, groups: list[list[int]]):
        wv = 0.0
        total = sum(len(group) for group in groups)
        for group in groups:
            wv += len(group)*self.variance(group)/total
        return wv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode=True

import numpy as np
from skhmn_learn.decision_trees.trees import DecisionTrees
from skhmn_learn.decision_trees.trees import Impurity

dt = DecisionTrees()

print(dt.gini_impurity(np.array([1,1,0,1,0])))
print(dt.gini_impurity(np.array([1,1,0,1,0,0])))
print(dt.gini_impurity(np.array([1,1,1,1])))

print(dt.entropy(np.array([1,1,0,1,0])))
print(dt.entropy(np.array([1,1,0,1,0,0])))
print(dt.entropy(np.array([1,1,1,1])))

print(dt.weighted_impurity([[1, 0, 1], [0, 1]], Impurity.ENTROPY))
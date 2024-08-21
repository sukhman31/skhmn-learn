import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode=True

import numpy as np
from skhmn_learn.decision_trees.trees import DecisionTrees

dt = DecisionTrees()

# print(dt.gini_impurity(np.array([1,1,0,1,0])))
# print(dt.gini_impurity(np.array([1,1,0,1,0,0])))
# print(dt.gini_impurity(np.array([1,1,1,1])))

# print(dt.entropy(np.array([1,1,0,1,0])))
# print(dt.entropy(np.array([1,1,0,1,0,0])))
# print(dt.entropy(np.array([1,1,1,1])))

# print(dt.weighted_impurity([[1, 0, 1], [0, 1]], Impurity.ENTROPY))

X_train = np.array([['tech', 'professional', 24],
['fashion', 'student', 20],
['fashion', 'professional', 25],
['sports', 'student', 18],
['tech', 'student', 18],
['tech', 'retired', 57],
['sports', 'professional', 36]])
y_train = np.array([1, 1, 0, 0, 1, 0, 1])

dt.train_tree(X_train, y_train)

# print("------------")
# print(dt.parent.left_child.training_data)
# print(dt.parent.right_child.training_data)
# print("------------")
# print(dt.parent.left_child.left_child.training_data)
# print(dt.parent.left_child.right_child.training_data)
# print("------------")
# print(dt.parent.right_child.left_child)
# print(dt.parent.right_child.right_child)

print(dt.classify(np.array(['tech', 'professional', 27])))

# print(dt.weighted_variance([[1,2,3],[1,2]]))

X_train_reg = np.array([['tech', 'professional', 24],
['fashion', 'student', 20],
['fashion', 'professional', 25],
['sports', 'student', 18],
['tech', 'student', 18],
['tech', 'retired', 57],
['sports', 'professional', 36]])
y_train_reg = np.array([15.6, 14.7, 17.8, 19.3, 14.67, 13.4, 16.8])

dt.train_regression_tree(X_train_reg, y_train_reg)
print(dt.evaluate(np.array(['tech', 'student', 57])))
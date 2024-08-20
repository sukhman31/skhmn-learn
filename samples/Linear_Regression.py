import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode=True

import numpy as np
from skhmn_learn.linear_regression.lin_reg import LinearRegression

lr = LinearRegression()

X_train = np.array([[6], [2], [3], [4], [1], [5], [2], [6], [4], [7]])
y_train = np.array([5.5, 1.6, 2.2, 3.7, 0.8, 5.2, 1.5, 5.3, 4.4, 6.8], ndmin=2, dtype=float).T

lr.train(X_train, y_train)
print(lr.evaluate(np.array([8])))
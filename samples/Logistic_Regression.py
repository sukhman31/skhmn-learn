import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode=True

import numpy as np
from skhmn_learn.logistic_regression.log_reg import LogisticRegression
from skhmn_learn.feature_extraction import count_vectorizer

lr = LogisticRegression()
vec = count_vectorizer.CountVectorizer()

corpus = ["Chinese Beijing Chinese","Chinese Chinese Shanghai","Chinese Macao","Tokyo Japan Chinese"]

vec.fit(corpus)

data = np.array(vec.fitMatrix)
label = np.array([1,1,1,-1], ndmin=2, dtype=float).T

lr.train(data, label, epochs=10)
print(lr.classify(np.array(vec.transform(["Tokyo Chinese"])), thresh=0.5))
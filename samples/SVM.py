import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode=True

from skhmn_learn.support_vector.svm import SVM
from skhmn_learn.feature_extraction import count_vectorizer
import numpy as np

svm = SVM()
vec = count_vectorizer.CountVectorizer()

corpus = ["Chinese Beijing Chinese","Chinese Chinese Shanghai","Chinese Macao","Tokyo Japan Chinese"]

vec.fit(corpus)

data = np.array(vec.fitMatrix)
label = np.array([1,1,1,-1], ndmin=2, dtype=float).T

svm.train(data,label)
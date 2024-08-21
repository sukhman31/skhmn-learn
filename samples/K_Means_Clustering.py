import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode=True

import numpy as np
from skhmn_learn.k_means.clustering import KMeansClustering

kmc = KMeansClustering()

kmc.train(np.array([[1,1], [2,3], [4,5], [1,-1], [2,4], [0,0]]), k = 2)
print(kmc.classify(np.array([[0.5,1], [3,4]])))
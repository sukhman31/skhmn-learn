import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode=True

import numpy as np
from skhmn_learn.feature_extraction.encoding import OneHotEncoding

ohe = OneHotEncoding()

print(ohe.fit_transform(np.array([['tech', 'professional'],
['fashion', 'student'],
['fashion', 'professional'],
['sports', 'student'],
['tech', 'student'],
['tech', 'retired'],
['sports', 'professional']])))
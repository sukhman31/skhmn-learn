import numpy as np
import pandas as pd

class OneHotEncoding:

    def __init__(self):
        self.dictionary = None

    def fit(self, content: np.ndarray):
        self.dictionary = []
        for i in range(content.shape[1]):
            column = content[:,i]
            self.dictionary.append(list(np.unique(column)))
    
    def transform(self, content: np.ndarray):
        final_val = []
        for entry in content:
            entry_val = []
            for idx, col in enumerate(entry):
                val = [0]*len(self.dictionary[idx])
                if col in self.dictionary[idx]:
                    val[self.dictionary[idx].index(col)] = 1
                entry_val.extend(val)
            final_val.append(entry_val)
        return np.array(final_val)

    def fit_transform(self, content: np.ndarray):
        self.fit(content)
        return self.transform(content)
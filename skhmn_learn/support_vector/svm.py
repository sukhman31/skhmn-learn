import numpy as np

class SVM:

    def __init__(self):
        self.reg_term = 1.0
        self.weights = []
        self.bias = []

    def train(self, training_data: np.ndarray, training_class: np.ndarray, epochs: int = 10, lr: float = 0.02):
        self.weights = np.random.randn(1,len(training_data[0])).T
        self.bias = np.zeros([len(training_data),1])
        for _ in range(epochs):
            y_calc = np.dot(training_data, self.weights) - self.bias
            print(self.calculate_loss(training_class, y_calc))
            for idx,data in enumerate(training_data):
                condition = training_class[idx]*y_calc[idx] >= 1
                if condition:
                    self.weights -= 2*self.reg_term*lr*self.weights
                else:
                    self.weights -= lr*(2*self.reg_term*self.weights - np.array([d*training_class[idx] for d in data]))
                    self.bias -= lr*training_class[idx]

    def calculate_loss(self, classification: np.ndarray, calculated_class: np.ndarray):
        return self.reg_term * self.get_weight_mod() + self.calculate_hinge_loss(classification, calculated_class)

    def get_weight_mod(self):
        return np.linalg.norm(self.weights)

    def calculate_hinge_loss(self, classification: np.ndarray, calculated_class: np.ndarray):
        hinge_func = np.vectorize(self.hinge_loss_func)
        hinge_loss_mat = hinge_func(np.multiply(classification, calculated_class))
        return float(np.sum(hinge_loss_mat)/len(hinge_loss_mat))
    
    def hinge_loss_func(self, x):
        return max(0,1-x)
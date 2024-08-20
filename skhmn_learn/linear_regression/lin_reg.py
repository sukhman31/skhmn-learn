import numpy as np

class LinearRegression:

    def __init__(self):
        self.weights = None
    
    def train(self, training_data: np.ndarray, training_class: np.ndarray, epochs: int = 10, lr: float = 0.02):
        self.weights = np.random.randn(len(training_data[0]),1)
        self.bias = np.zeros([len(training_data),1])
        for _ in range(epochs):
            y_calc = np.dot(training_data, self.weights)
            print(self.calculate_loss(y_calc, training_class))
            self.weights += lr*np.dot(training_data.T, training_class - y_calc)/len(training_class)
    
    def evaluate(self, eval_data: np.ndarray):
        return np.dot(eval_data, self.weights)

    def calculate_loss(self, classification: np.ndarray, calculated_class: np.ndarray):
        sum = 0.0
        for exp, calc in zip(classification, calculated_class):
            sum += (exp-calc)**2
        return (sum/len(classification))[0]
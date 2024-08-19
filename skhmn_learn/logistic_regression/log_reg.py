import numpy as np

class LogisticRegression:

    def __init__(self):
        self.weights = None
    
    def train(self, training_data: np.ndarray, training_class: np.ndarray, epochs: int = 10, lr: float = 0.05):
        self.weights = np.random.randn(len(training_data[0]),1)
        for _ in range(epochs):
            y_calc = self.sigmoid(training_data)
            print(self.calculate_loss(training_class, y_calc))
            self.weights -= lr*np.dot(training_data.T, training_class - y_calc)/len(training_class)

    def classify(self, eval_data: np.ndarray, thresh: float):
        calc = self.sigmoid(eval_data)
        if calc < thresh:
            return 0
        return 1

    def calculate_loss(self, classification: np.ndarray, calculcated_class: np.ndarray):
        return -1*(np.sum(np.multiply(classification, np.log(calculcated_class))) + np.sum(np.multiply(1-classification, np.log(1-calculcated_class)))) / len(calculcated_class)
    
    def sigmoid(self, training_data: np.ndarray):
        return 1.0 / (1 + np.exp(np.dot(training_data, self.weights)))
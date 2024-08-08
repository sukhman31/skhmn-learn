#TODO: Currently only supporting binary classification, will later on support multiple classes

class NaiveBayes:

    def __init__(self, laplace_smoothing_factor=None):
        self.class_prob = []
        self.laplace_smoothing_factor = laplace_smoothing_factor
        self.positive_class_dictionary = {}
        self.negative_class_dictionary = {}

    def train(self, training_data: list[list[int]], training_class: list[int]):
        self.__calculate_class_prob(training_class)
        self.__calculate_class_dictionaries(training_data, training_class)

    def calculate_prob(self, eval_data: list[int]):
        prob = [1,1]
        for data in eval_data:
            prob[0] *= self.__get_likelihood_for_negative_class(data)
            prob[1] *= self.__get_likelihood_for_positive_class(data)
        prob = [p/self.class_prob[i] for i,p in enumerate(prob)]
        sum_prob = sum(prob)
        return [p/sum_prob for p in prob]


    def __get_likelihood_for_negative_class(self, data: int) -> float:
        return self.negative_class_dictionary.get(data,0)+1 / sum(self.negative_class_dictionary.values()) + len(self.negative_class_dictionary.keys())
    
    def __get_likelihood_for_positive_class(self, data: int) -> float:
        return self.positive_class_dictionary.get(data,0)+1 / sum(self.positive_class_dictionary.values()) + len(self.positive_class_dictionary.keys())

    def __calculate_class_dictionaries(self, training_data: list[list[int]], training_class: list[int]):
        for data, cl in zip(training_data, training_class):
            if cl==0:
                for entry in data:
                    for entity in entry:
                        self.negative_class_dictionary[entity] = self.negative_class_dictionary.get(entity,0) + 1
            else:
                for entry in data:
                    for entity in entry:
                        self.positive_class_dictionary[entity] = self.positive_class_dictionary.get(entity,0) + 1

    def __calculate_class_prob(self, training_class: list[int]):
        class_prob = [0,0]
        for cl in training_class:
            class_prob[cl] += 1
        self.class_prob = [prob/len(class_prob) for prob in class_prob]

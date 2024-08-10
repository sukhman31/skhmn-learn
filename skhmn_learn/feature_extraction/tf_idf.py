from skhmn_learn.feature_extraction.count_vectorizer import CountVectorizer
import math

class TfIdf:

    def __init__(self):
        self.__count_vectorizer = CountVectorizer()
        self.fitMatrix = []
        self.idf = []
    
    def fit(self, content: list[str]):
        self.__count_vectorizer.fit(content)
        self.fitMatrix = self.__count_vectorizer.fitMatrix
        for idx in range(len(self.fitMatrix[0])):
            self.idf.append(math.log10(len(content)/(1+len(self.__count_vectorizer.documentOccurence.get(idx)))))
        for idx, line in enumerate(self.fitMatrix):
            for idx1, value in enumerate(line):
                self.fitMatrix[idx][idx1] = value * self.idf[idx1]
    
    def print(self):
        print(self.__count_vectorizer.dictionary)
        print(self.__count_vectorizer.documentOccurence)
        print(self.__count_vectorizer.countOfWords)
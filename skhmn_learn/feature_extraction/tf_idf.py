from skhmn_learn.feature_extraction.count_vectorizer import CountVectorizer
import math

class TfIdf:

    def __init__(self):
        self.__count_vectorizer = CountVectorizer()
        self.fitMatrix = None
    
    def fit(self, content: list[str]):
        self.__count_vectorizer.fit(content)
        self.fitMatrix = self.__count_vectorizer.fitMatrix
        idf = []
        for idx in range(len(self.fitMatrix[0])):
            idf.append(math.log10(len(content)/(1+len(self.__count_vectorizer.documentOccurence.get(idx)))))
        for idx, line in enumerate(self.fitMatrix):
            for idx1, value in enumerate(line):
                if value != 0:
                    self.fitMatrix[idx][idx1] = value * idf[idx1]
    
    def transform(self, content: list[str]):
        fitMatrix = self.__count_vectorizer.transform(content)
        idf = []
        for idx in range(len(fitMatrix[0])):
            idf.append(math.log10(len(content)/(1+len(self.__count_vectorizer.documentOccurence.get(idx)))))
        print(idf)
        print(fitMatrix)
        for idx, line in enumerate(fitMatrix):
            for idx1, value in enumerate(line):
                if value != 0:
                    fitMatrix[idx][idx1] = value * idf[idx1]
        
        return fitMatrix

    def fit_transform(self, content: list[str]):
        self.fit(content)
        return self.transform(content)
    
    def print(self):
        print(self.__count_vectorizer.dictionary)
        print(self.__count_vectorizer.documentOccurence)
        print(self.__count_vectorizer.countOfWords)
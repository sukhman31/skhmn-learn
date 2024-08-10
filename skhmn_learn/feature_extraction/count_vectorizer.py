# TODO: convert fitMatrix to sparse matrix (possibly make a class sparse matrix and then add toArray function to it?) 

class CountVectorizer:

    def __init__(self):
        self.dictionary = []
        self.countOfWords = {}
        self.fitMatrix = []
    
    def fit(self, content: list[str]):
        self.countOfWords = {}
        self.dictionary = []
        for line in content:
            for word in line.split(" "):
                self.countOfWords[word] = 1 + self.countOfWords.get(word,0)
                if word not in self.dictionary:
                    self.dictionary.append(word)
        self.fitMatrix = [[0 for word in self.dictionary] for line in content]
        for i, line in enumerate(content):
            for word in line.split(" "):
                self.fitMatrix[i][self.dictionary.index(word)] += 1
    
    def transform(self, content: list[str]) -> list[list[int]]:
        fitMatrix = [[0 for word in self.dictionary] for line in content]
        for i, line in enumerate(content):
            for word in line.split(" "):
                if word in self.dictionary:
                    fitMatrix[i][self.dictionary.index(word)] += 1
        return fitMatrix
    
    def fit_transform(self, content: list[str]) -> list[list[int]]:
        self.fit(content)
        return self.transform(content)
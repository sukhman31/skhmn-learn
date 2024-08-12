import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode=True

from skhmn_learn.feature_extraction import count_vectorizer
from skhmn_learn.naive_bayes import nb

vec = count_vectorizer.CountVectorizer()
nb = nb.NaiveBayes()

corpus = ["Chinese Beijing Chinese","Chinese Chinese Shanghai","Chinese Macao","Tokyo Japan Chinese"]

vec.fit(corpus)

print(vec.countOfWords)
print(vec.dictionary)
print(vec.fitMatrix)
print(vec.documentOccurence)

data = vec.fitMatrix
label = [1,1,1,0]

nb.train(data, label)

test_data = vec.transform(["Chinese Chinese Chinese Tokyo Japan"])
print("test_data", test_data)
print(nb.calculate_prob(test_data[0]))
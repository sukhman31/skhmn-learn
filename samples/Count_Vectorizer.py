import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode=True

from skhmn_learn.feature_extraction import count_vectorizer

vec = count_vectorizer.CountVectorizer()

corpus = ["Chinese Beijing Chinese","Chinese Chinese Shanghai","Chinese Macao","Tokyo Japan Chinese"]

vec.fit(corpus)

print(vec.countOfWords)
print(vec.dictionary)
print(vec.fitMatrix)
print(vec.documentOccurence)

test_data = vec.transform(["Chinese Chinese Chinese Tokyo Japan"])
print("test_data", test_data)
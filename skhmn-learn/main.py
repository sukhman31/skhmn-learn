from feature_extraction import count_vectorizer

vec = count_vectorizer.CountVectorizer()

corpus = ["this is a file","This is another file","Here is a document","Bring forth another document"]

vec.fit(corpus)

print(vec.countOfWords)
print(vec.dictionary)
print(vec.fitMatrix)
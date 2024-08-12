import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode=True

from skhmn_learn.feature_extraction import tf_idf

ti = tf_idf.TfIdf()

corpus = ["Air quality in the sunny island improved gradually throughout Wednesday","Air quality in Singapore on Wednesday continued to get worse as haze hit the island","The air quality in Singapore is monitored through a network of air monitoring stations located in different parts of the island","The air quality in Singapore got worse on Wednesday"]

ti.fit(corpus)

# print(ti.fitMatrix)
print(ti.transform(["Air on Wednesday in Singapore is monitored"]))
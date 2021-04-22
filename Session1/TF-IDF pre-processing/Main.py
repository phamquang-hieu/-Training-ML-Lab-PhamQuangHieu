from ComputeTFIDF import generateVocabulary, get_tf_idf
from GatherData import *

if __name__ == '__main__':
	data_path = "datasets/20news-bydate/20news-full-processed.txt"
	gather20NewsGroupsData()
	generateVocabulary(data_path)
	get_tf_idf(data_path)

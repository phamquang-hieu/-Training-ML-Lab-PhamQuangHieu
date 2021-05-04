import numpy as np
import operator
from collections import defaultdict
import sys
def generateVocabulary(data_path):
	def compute_idf(df, corpusSize):
		assert df > 0 
		return np.log10(corpusSize * 1. /df)

	with open(data_path) as f:
		lines = f.read().splitlines() # list of corpuses
	docCount =	defaultdict(int) # this is to prevent error in the command: docCount[word] +=1 
	corpusSize = len(lines)

	for line in lines:
		# each line is equivalent to a document
		features = line.split('<fff>') # [label , filename, text]
		text = features[-1]
		words = list(set(text.split()))
		for word in words:
			docCount[word] +=1
	wordsIdfs = [(word, compute_idf(documentFreq, corpusSize))
				for word, documentFreq in docCount.items()
				if documentFreq > 10 and not word.isdigit()]
	wordsIdfs.sort(key=operator.itemgetter(1), reverse=True)
	print("Vocabulary size: {}".format(len(wordsIdfs)))
	with open("datasets/20news-bydate/words_idfs.txt", "w") as f:
		f.write('\n'.join([word + '<fff>' + str(idf) for word, idf in wordsIdfs]))


def get_tf_idf(data_path):
	with open("datasets/20news-bydate/words_idfs.txt") as f:
		wordsIdfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1]))
					for line in f.read().splitlines()]
		wordIds = dict([word, index] for index, (word, idf) in enumerate(wordsIdfs))

		idfs = dict(wordsIdfs)

	with open(data_path) as f:
		documents = [
			(int(line.split('<fff>')[0]),
			int(line.split('<fff>')[1]),
			line.split('<fff>')[2])
			for line in f.read().splitlines()
		]

	data_tf_idf = []

	for document in documents:
		label, doc_id, text = document
		words = [word for word in text.split() if word in idfs.keys()]
		# print(len(words))
		# sys.exit()
		word_set = list(set(words))
		max_term_freq = max([words.count(word) for word in word_set])

		words_tfidfs = []
		sum_squares = 0.0

		for word in word_set:
			term_freq = words.count(word)
			tf_idf_value = term_freq * 1. /max_term_freq * idfs[word]
			words_tfidfs.append((wordIds[word], tf_idf_value))
			sum_squares += tf_idf_value**2

		words_tfidfs_normalized = [str(index) + ':'
									+ str(tf_idf_value/ np.sqrt(sum_squares))
									for index, tf_idf_value in words_tfidfs]
		sparse_rep = ' '.join(words_tfidfs_normalized)
		data_tf_idf.append((label, doc_id, sparse_rep))
	with open("datasets/20news-bydate/20news-full-tfidf.txt", "w") as f:
		f.write('\n'.join(str(line[0]) + '<fff>' + str(line[1]) + '<fff>' + str(line[2])  for line in data_tf_idf) )

# generateVocabulary("datasets/20news-bydate/20news-full-processed.txt")
# get_tf_idf("datasets/20news-bydate/20news-full-processed.txt")
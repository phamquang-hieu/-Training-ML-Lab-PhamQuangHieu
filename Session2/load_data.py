import numpy as np

def load_data(data_path):
	def sparse_to_dense(sparse_r_d, vocab_size):
		"""the purpose is to give the tf-idf vector representation for document d"""
		# sparse_r_d (a string): sparse tf-idf representation of document d
		r_d = [0.0 for _ in range(vocab_size)]
		indices_tfidfs = sparse_r_d.split() 
		for index_tfidf in indices_tfidfs:
			index = int(index_tfidf.split(':')[0])
			tfidf = float(index_tfidf.split(':')[1])
			r_d[index] = tfidf
		return np.array(r_d)
	
	# data_path is the path to the tf_idf sparse representation file.
	with open(data_path) as f:
		d_lines = f.read().splitlines()
	with open('datasets/20news-bydate/words_idfs.txt') as f:
		# vocab_size = size of our dictionary
		vocab_size = len(f.read().splitlines())

	data = []
	labels = []

	for data_id, d in enumerate(d_lines):
		# the structure of each line is [label_id]<fff>[documentID]<fff>[[word_id]:[tf_idf]]
		features = d.split('<fff>')
		# each d corespond to a line in the data set

		label, doc_id = int(features[0]), int(features[1])
		labels.append(label)
		r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
		data.append(r_d)
	return data, labels
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import *
from load_data import load_data


def compute_purity(self):
		majority_sum = 0
		for cluster in self._clusters:
			member_labels = [member._label for member in cluster._members]
			max_count = max([member_labels.count(label) for label in range(20)])
			majority_sum += max_count
		return majority_sum* 1/ len(self._data)

def clustering_with_KMeans():
	data, original_label = load_data(data_path= 'datasets/20news-bydate/words_tfidfs_rep.txt')
	
	X = csr_matrix(data)
	print ('------------')
	kmeans = KMeans(
		n_clusters=40,
		init='k-means++',
		n_init=5,
		tol=1e-3,
		random_state=2018
	).fit(X)

	labels = kmeans.labels_
	with open("kMeansSklearn_out.txt", "w") as f:
		for label in labels:
			f.write(str(label))
			f.write(" ")

	with open("kMeansSklearn_original_label.txt", "w") as f:
		for label in original_label:
			f.write(str(label))
			f.write(" ")

clustering_with_KMeans()
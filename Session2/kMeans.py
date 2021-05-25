from random import uniform, randint
import numpy as np
from collections import defaultdict
import sys
from time import time

class Member:
	def __init__(self, r_d, label=None, doc_id=None):
		self._r_d = r_d
		self._label = label
		self._doc_id = doc_id

class Cluster:
	def __init__(self):
		self._centroid = None
		self._members = []

	def reset_menebers(self):
		self._members = []

	def add_member(self, member):
		self._members.append(member)

class Kmeans:
	def __init__(self, num_clusters):
		self._num_clusters = num_clusters
		self._clusters = [Cluster() for _ in range(self._num_clusters)]
		self._E = [] # list of initial centroids
		self._S = 0 # overall similarity


	def load_data(self, data_path):
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

		self._data = []

		# counting the number of instances in each label (used for MMI function).
		self._label_count = defaultdict(int)
		for data_id, d in enumerate(d_lines):
			# the structure of each line is [label_id]<fff>[documentID]<fff>[[word_id]:[tf_idf]]
			features = d.split('<fff>')
			# each d corespond to a line in the data set

			label, doc_id = int(features[0]), int(features[1])
			self._label_count[label] += 1
			r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
			self._data.append(Member(r_d = r_d, label = label, doc_id=doc_id))

	def compute_similarity(self, member, centroid):
		# the centroid's norm has been normalized to 1 in the function update_centroid_of
		# member's norm has been normalized when we compute the TFIDF representation.
		# print("enter compute_similarity")
		return member._r_d.dot(centroid)
	

	def random_init(self, seed_value):
		# randomly select k centroids to put in self._E
		# seed value is the first centroid to be chosen.
		starttime = time()
		self._E.append(seed_value)

		added_centroid_num = 1 
		hash_table = 0
		while added_centroid_num < self._num_clusters:
			new_centroid = None
			min_similarity_val = 1
			candidate_centroid_id = None
			new_centroid_id = None
			for i in range(len(self._data)):
				candidate_centroid = None
				local_max_similarity = -1
				if (1<<i) & hash_table == 0:
					for centroid in self._E:
						tmp = self.compute_similarity(self._data[i], centroid._r_d)
						if tmp > local_max_similarity:
							local_max_similarity = tmp
							candidate_centroid = self._data[i]
							candidate_centroid_id = i

					if local_max_similarity < min_similarity_val:
						min_similarity_val = local_max_similarity
						# print("min_similarity_val ",min_similarity_val)
						new_centroid = candidate_centroid
						new_centroid_id = candidate_centroid_id

			if new_centroid:
				self._E.append(new_centroid)
				# print("new_centroid_id:", new_centroid_id)
				hash_table = hash_table |(1<<new_centroid_id)
				added_centroid_num += 1
			else:
				print("catch")
			

		for i in range(len(self._clusters)):
			self._clusters[i]._centroid = self._E[i]._r_d 
		endtime = time()
		print("execution time: ", endtime -starttime)


	def select_cluster_for(self, member):
		# select appropriate cluster for 1 member and return the maximum number of simirarity
		# i.e the greatest cosine similarity denote the most similar.
		# print("enter select_cluster_for")
		best_fit_cluster = None
		max_similarity = -1
		for cluster in self._clusters:
			similarity = self.compute_similarity(member, cluster._centroid)
			if similarity > max_similarity:
				best_fit_cluster = cluster
				max_similarity = similarity

		best_fit_cluster.add_member(member)
		return max_similarity

	def update_centroid_of(self, cluster):
		# print("enter update_centroid_of",)
		member_r_ds = [member._r_d for member in cluster._members]

		# print("member_r_ds: ", member_r_ds)

		aver_r_d = np.mean(member_r_ds, axis=0)
		# print("aver_r_d: ", aver_r_d)
		
		sqrt_sum_sqr = np.sqrt(np.sum(aver_r_d**2))
		new_centroid = np.array([value / sqrt_sum_sqr for value in aver_r_d])
		# because we are estimating the similarity by cosine, normalizing new_centroid make it easier to compute the cosine.
		cluster._centroid = new_centroid

	def stopping_condition(self, criterion, threshold):
		# print("enter stopping_condition")
		criteria = ['centroid', 'similarity', 'max_iters']
		assert criterion in criteria

		# threshold = number of iterations
		if criterion == 'max_iters':
			if self._iteration >= threshold:
				return True
			return False

		#threshhold = number of cluster which have changes in centroid.
		elif criterion == 'centroid':
			E_new = [list(cluster._centroid) for cluster in self._clusters]
			E_new_minus_E = [centroid for centroid in E_new if centroid not in self._E]
			self._E = E_new
			if len(E_new_minus_E) <= threshold:
				return True
			return False
		#threshold = total number of similarity between 2 iterations
		else:
			new_S_minus_S = self._new_S - self._S
			self._S = self._new_S
			if new_S_minus_S <= threshold:
				return True
			return False


	def run(self, seed_value, criterion, threshold):
		self.random_init(seed_value)
		# continually update cluster until convergence
		self._iteration = 0

		while True:
			# reset clusters, retain only centroids
			for cluster in self._clusters:
				cluster.reset_menebers()
			# new total number of similarity.
			# this value is taken into consideration if we choose to compare the total value of
			# similarity between 2 iterations.
			self._new_S = 0 

			for member in self._data:
				max_s = self.select_cluster_for(member)
				self._new_S += max_s

			for cluster in self._clusters:
				# print("cluster length: ", len(cluster._members))
				self.update_centroid_of(cluster)

			self._iteration += 1
			if self.stopping_condition(criterion, threshold):
				break

	def compute_purity(self):
		majority_sum = 0
		for cluster in self._clusters:
			member_labels = [member._label for member in cluster._members]
			max_count = max([member_labels.count(label) for label in range(20)])
			majority_sum += max_count
		return majority_sum* 1/ len(self._data)

	def compute_MMI(self):
		# H(): entropy function
		# I value = H(C) - H(C|W)
		# H(omega): cluster entropy
		# H(C): class entropy
		# N: number of members
		I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)
		for cluster in self._clusters:
			wk = len(cluster._members) *1.
			H_omega += -wk / N * np.log10(wk / N)
			member_labels = [member._label 
							for member in cluster._members]

			for label in range(20):
				wk_cj = member_lables.count(label) * 1.

				cj = self._label_count[label]

				I_value += wk_cj/N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
				# the reason for adding 1e-12 is because of our convention 0log10(0) = 0
		for label in range(20):
			cj = self._label_count[label]/N
			H_C += -cj / N * log10(cj / N)

		return I_value * 2. /(H_C + H_omega)

if __name__ == '__main__':
	k = Kmeans(30)
	print(k._num_clusters)
	k.load_data('datasets/20news-bydate/words_tfidfs_rep.txt')
	seed_value = k._data[randint(0, len(k._data)-1)]

	k.run(seed_value=seed_value, criterion='centroid', threshold=2)
	print(k.compute_purity())

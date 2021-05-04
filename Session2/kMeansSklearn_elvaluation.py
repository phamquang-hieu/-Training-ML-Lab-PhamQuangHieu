import numpy as np
with open("kMeansSklearn_out.txt", "r") as f:
	for line in f:
		pred_label = list(map(int, line.split()))

with open("kMeansSklearn_original_label.txt", "r") as f:
	for line in f:
		true_label = list(map(int, line.split()))

def compute_purity(pred_label, true_label):
	mat = [[0 for _ in range(int(max(true_label))+1)] for _ in range(int(max(pred_label))+1)]
	for i in range(len(pred_label)):
		mat[int(pred_label[i])][int(true_label[i])] += 1
	total = 0
	mat = np.array(mat)
	for line in mat:
		total += np.max(line)/np.sum(line)
	print(total*1.0/len(mat))

compute_purity(pred_label, true_label)
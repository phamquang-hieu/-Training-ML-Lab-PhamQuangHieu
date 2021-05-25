from load_data import load_data
import numpy as np

def compute_accuracy(predicted_y, expected_y):
	matches = np.equal(predicted_y, expected_y)
	accuracy = np.sum(matches.astype(float))/len(expected_y)
	return accuracy

def classifying_with_linear_SVMs():
	train_X, train_Y = load_data("datasets/20news-bydate/20news-train-tfidf.txt")
	from sklearn.svm import LinearSVC, SVC

	classifier = SVC(
		C=50.0, # penalty coeff
		kernel='rbf',
		gamma=0.1,
		tol= 0.001, # tolerance for stopping criteria
		verbose=True # True: log, False: no log
	)

	classifier.fit(train_X, train_Y)

	test_X, test_Y = load_data(data_path="datasets/20news-bydate/20news-test-tfidf.txt")
	predicted_y = classifier.predict(test_X)
	accuracy = compute_accuracy(predicted_y=predicted_y, expected_y =test_Y)
	print("Accuracy: ", accuracy)

classifying_with_linear_SVMs()
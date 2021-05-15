import numpy as np 
from random import shuffle


def normalize_and_add_ones(dat):
	X = np.array(dat)
	X_max = np.array([ [np.amax(X[:, column]) for column in range(X.shape[1])] for _ in range(X.shape[0])])
	X_min = np.array([ [np.amin(X[:, column]) for column in range(X.shape[1])] for _ in range(X.shape[0])])
	normalized_X = (X[:, :] - X_min)/(X_max - X_min)
	return np.column_stack(([1 for _ in range(X.shape[0])], normalized_X))


class RidgeRegression:
	def __init__(self):
		return;
	def computeRss(self, Y_new, Y_train):
		loss = 1. / Y_new.shape[0] * \
				np.sum((Y_new - Y_train)**2)
		return loss
	def predict(self, W, X_new):
		X_new = np.array(X_new)
		result = X_new.dot(W)
		return result
	def fit(self, X_train, Y_train, LAMBDA):
		"""train the model by OLS"""
		assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]

		w = np.linalg.inv( (X_train.T).dot(X_train) + LAMBDA*np.identity(X_train.shape[1]) ).dot(X_train.T.dot(Y_train))
		return w	

	def fit_gradient_descent(self, X_train, Y_train, LAMBDA, learning_rate, epoch_num = 1000, batch_size = 20):
		"""Train the model by using gradient descent"""

		# inititalize w and last lost = +oo
		w = np.random.randn(X_train.shape[1])
		lastLost = 10e+8
		for epoch in range(epoch_num):
			# Finding the best training order
			training_order = np.array([x for x in range(X_train.shape[0])])
			shuffle(training_order)
			X_train = X_train[training_order]
			Y_train = Y_train[training_order]
			
			# perform mini batch update to prevent local extremenum
			miniBatchNumb = int(np.ceil(X_train.shape[0] / batch_size))
			for batch in range(miniBatchNumb):
				startIndex = batch*(batch_size)
				xTrainBatch = X_train[startIndex:startIndex + batch_size, :]
				yTrainBatch = Y_train[startIndex:startIndex + batch_size]
				
				grad = xTrainBatch.T.dot(xTrainBatch.dot(w) - yTrainBatch)* 1/(X_train.shape[1])  + LAMBDA*w
				w = w - learning_rate*grad
			new_loss = self.computeRss(self.predict(w,X_train), Y_train)
			if(np.abs(new_loss - lastLost) <= 1e-5):
				break
			lastLost = new_loss
		return w

	def get_the_best_LAMBDA(self, X_train, Y_train):
		def cross_validation(num_folds, LAMBDA):
			"""this function will return the average RSS over the whole trainning set with corresponding LAMBDA value by using cross validation"""
			row_ids = np.array(range(X_train.shape[0]))
			# devide the trainning set in to folds
			# each fold contains (len(row_ids) - len(row_ids) % num_folds)/num_folds
			# the remaining training instance is pushed in to the last fold to prevent error in the np.split() function
			valid_ids = np.split(row_ids[:len(row_ids) - len(row_ids) % num_folds], num_folds)
			valid_ids[-1] = np.append(valid_ids[-1], row_ids[len(row_ids - len(row_ids) % num_folds):])
			train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
			total_RSS = 0
			for i in range(num_folds):
				train_part = {'X': X_train[train_ids[i]], 'Y': Y_train[train_ids[i]]}
				valid_part = {'X': X_train[valid_ids[i]], 'Y': Y_train[valid_ids[i]]}
				W = self.fit(train_part['X'], train_part['Y'], LAMBDA)
				Y_predicted = self.predict(W, valid_part['X'])
				total_RSS += self.computeRss(Y_predicted, valid_part['Y'])
			return total_RSS / num_folds


		def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values):
			"""scan for the most suitable LAMBDA value in a specific range with predefined step between two values"""
			for current_LAMBDA in LAMBDA_values:
				aver_RSS = cross_validation(num_folds = 5, LAMBDA = current_LAMBDA)
				if aver_RSS < minimum_RSS:
					best_LAMBDA = current_LAMBDA
					minimum_RSS = aver_RSS
			return best_LAMBDA, minimum_RSS


		# First scan: roughly find an integer LAMBDA value
		best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=0, minimum_RSS=10000 **2, LAMBDA_values= range(50))

		# Second scan: calibrate the values found in previous step a little bit.

		LAMBDA_values = [k * 1. /1000 for k in range( max(0, (best_LAMBDA -1) * 1000), max(1, best_LAMBDA + 1)*1000) ]
		best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=best_LAMBDA, minimum_RSS=minimum_RSS, LAMBDA_values=LAMBDA_values)
		print("best Lambda is:", best_LAMBDA)
		return best_LAMBDA


							
if __name__ == '__main__':
	data = []
	with open("deathRate.txt", "r") as f:
		for line in f: 
			line = list(map(float, line.split()))
			data.append(line[1:])
	data = np.array(data)
	X = np.array(data[:, :len(data[0])-1]) # take all except the last column for Y
	Y = np.array(data[:, -1])
	X = normalize_and_add_ones(X)
	X_train, Y_train = X[: 50], Y[: 50]
	print(X_train.shape, Y_train.shape)
	X_test, Y_test = X[50:], Y[50:]
	print(X_test.shape, Y_test.shape)

	r = RidgeRegression()
	LAMBDA = r.get_the_best_LAMBDA(X_train, Y_train)
	print(LAMBDA)
	W = r.fit(X_train, Y_train, LAMBDA)
	print(W.shape)
	# W2 = r.fit_gradient_descent(X_train, Y_train, LAMBDA, 0.01)
	# a = r.computeRss(r.predict(W, X_test), Y_test)
	# b = r.computeRss(r.predict(W2, X_test), Y_test)
	# print("total lost when using usual RSS", a)
	# print("total lost when using fit_gradient_descent ", b)
	# print(a<b)

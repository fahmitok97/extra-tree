from tree import ExtraTreeClassifier as ETClassifier

class ExtraTreesClassifier():
	'''
	Instantiation should maintain any hyperparameter tuning for the Ensemble and ExtraTree itself
	Parameters:
	- num_trees = number of tree in the ensemble
	- max_features = maximum feature in single tree
	- min_split = minimum sample size for splitting
	'''
	def __init__(self, num_trees=200, max_features=10, min_split=2):
		self.num_trees = num_trees
		self.ensemble = [ETClassifier(max_features, min_split) * num_trees] #TODO: belum pasti parameter ET

	'''
	Train should implement the subroutine for training all ET tree in ensemble
	Input:
	- data : NxM training dataset with "pandas dataframe"-like format (i.e: dict of list)
	- target : Nx1 label for each training datapoint
	Output:
	- prediction: Nx1 output from supplying training data back to model
	- score: percentage of (correct_label/training_size)
	'''
	def train(self, data, target):
		predictions = [[target[0] * len(data[data.keys()[0]])] * self.num_trees]

		for idx in range(self.num_trees):
			predictions[idx] = self.ensemble[idx].train(data, target)

		prediction = [target[0] * len(data[data.keys()[0]])]
		num_correct_label = 0

		for idx_train in range(len(data[data.keys()[0]])):
			cnt = {}

			for idx_tree in range(self.num_trees):
				if predictions[idx_tree][idx_train] in cnt.keys():
					cnt[predictions[idx_tree][idx_train]] += 1
				else :
					cnt[predictions[idx_tree][idx_train]] = 1

			num_predicted = 0
			for pred in cnt.keys():
				if num_predicted < cnt[pred]:
					prediction[idx_train] = pred
					num_predicted = cnt[pred]

			if prediction[idx_train] == target[idx_train]:
				num_correct_label += 1

		return prediction, num_correct_label/len(data[data.keys()[0]])

	'''
	Predict should return predicted class from given dataset
	Input:
	- data:  NxM test dataset with "pandas dataframe"-like format (i.e: dict of list)
	Output:
	- prediction : Nx1 vector of predicted class for each datapoint
	'''
	def predict(self, data):
		predictions = [[] * self.num_trees]

		for idx in range(self.num_trees):
			predictions[idx] = self.ensemble[idx].predict(data)

		prediction = ["" * len(data[data.keys()[0]])]

		for idx_test in range(len(data[data.keys()[0]])):
			cnt = {}

			for idx_tree in range(self.num_trees):
				if predictions[idx_tree][idx_test] in cnt.keys():
					cnt[predictions[idx_tree][idx_test]] += 1
				else :
					cnt[predictions[idx_tree][idx_test]] = 1

			num_predicted = 0
			for pred in cnt.keys():
				if num_predicted < cnt[pred]:
					prediction[idx_test] = pred
					num_predicted = cnt[pred]

		return prediction
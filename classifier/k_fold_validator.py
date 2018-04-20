from random import randrange
from sklearn import metrics 
import numpy as np 

class CrossValidation: 

	def train_test_split(dataset, split=0.6): 
		train = list()
		train_size = split*len(dataset)
		dataset_copy = list(dataset)
		while len(train) < train_size:
			index = randrange(len(dataset_copy))
			train.append(dataset_copy.pop(index))
		return train, dataset_copy


	def cross_validation_split(dataset, k=10):
		dataset_split = list()
		dataset_copy = list(dataset)
		fold_size = int(len(dataset) / k)
		for i in range(fold): 
			fold = list()
			while len(fold) < fold_size: 
				index = randrange(len(dataset_copy))
				fold.append(dataset_copy.pop(index))
			dataset_split.append(fold)
		return dataset_split


	def cross_validation_fold(split, fold):
		training_set = list()
		validation_set = list()
		for i in range(len(split)): 
			if (i == fold):
				validation_set.append(split[i])
			else: 
				training_set.append(split[i])
		return training_set, validation_set


	def cross_validate(learner, k, dataset, label): 
		train_folds_score = []
		validation_folds_score = []
		for fold in range(0, k): 
			training_set, validation_set = cross_validation_split(dataset, k)
			training_label, validation_label = cross_validation_split(label, k)
			learner.fit(training_set, training_label)
			training_predict = learner.predict(training_set)
			validation_predict = learner.predict(validation_set)
			train_folds_score.append(metrics.accuracy_score(training_label, training_predict))
			validation_folds_score.append(metrics.accuracy_score(validation_label, validation_predict))
		return train_folds_score, validation_folds_score
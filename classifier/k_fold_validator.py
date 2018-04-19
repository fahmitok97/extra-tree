from random import randrange

def train_test_split(dataset, split=0.6): 
	train = list()
	train_size = split*len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy


def cross_validation_split(dataset, fold=10):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / fold)
	for i in range(fold): 
		fold = list()
		while len(fold) < fold_size: 
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split


def cross_validate():
	#TODO
	pass
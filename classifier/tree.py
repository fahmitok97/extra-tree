import math

def __entropy(labels):
	n = len(labels)
	unique_labels = list(set(labels))

	entropy = 0
	for label in unique_labels:
		cnt = 0
		for elem in labels:
			if elem == label:
				cnt += 1

		entropy -= (cnt/n)*math.log2(cnt/n)

	return entropy


def __score():
	pass

def __random_split():
	pass

class __ExtraTreeClassifier():
	## TODO: core implementation of ETClassifier
	pass
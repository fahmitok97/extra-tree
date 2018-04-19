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

def __max_freq(labels):
	n = len(labels)
	map_labels = {}

	for label in labels:
		if label in map_labels.keys():
			map_labels[label] += 1
		else:
			map_labels[label] = 1

	frequent_label = labels[0]
	freq = -1
	for label in map_labels.keys():
		if map_labels[label] > freq:
			freq = map_labels[label]
			frequent_label = label

	return frequent_label

def __score(labels, split):
	n = len(split)
	nAtt = len(split[0])

	h_c = __entropy(labels)
	C_ct = list()

	for i in range(nAtt) :
		curr_split = split[:,i]
		h_t = __entropy(curr_split)
		idx_c1 = [i for (i, val) in enumerate(curr_split) if val in curr_split]
		idx_c2 = [i for (i, val) in enumerate(curr_split) if val not in curr_split]
		y_c1 = labels[idx_c1]
		y_c2 = labels[idx_c2]

		h_ct = __entropy(y_c1) + __entropy(y_c2)
		I_ct = h_c - h_ct
		C_ct = 2*I_ct/(h_c+h_t)
	
	scores = C_ct
	
	return [val for (i, val) in enumerate(curr_split) if i == None]


def __random_split():
	pass

class __ExtraTreeClassifier():
	## TODO: core implementation of ETClassifier
	pass
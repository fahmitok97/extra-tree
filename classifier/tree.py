import math
import random

def _entropy(labels):
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

def _max_freq(labels):
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

def _score(labels, split):
	num_row = len(split[list(split.keys())[0]])
	num_feature = len(split.keys())

	out_entropy = _entropy(labels)
	normalized_info_gain = {}

	for feature in split.keys() :
		curr_split = split[feature]
		split_entropy = _entropy(curr_split)
		idx_left_child = [key for (key, val) in enumerate(curr_split) if val == True]
		idx_right_child = [key for (key, val) in enumerate(curr_split) if val == False]

		y_left_child = [ labels[key] for key in idx_left_child ]
		y_right_child = [ labels[key] for key in idx_right_child ]

		average_split_entropy = (len(y_left_child)/num_row)*_entropy(y_left_child) + (len(y_right_child)/num_row)*_entropy(y_right_child)
		info_gain = out_entropy - average_split_entropy
		normalized_info_gain[feature] = 2.0*info_gain/(out_entropy+split_entropy)

	return normalized_info_gain

def _random_split(dataset):
	num_row = len(dataset[list(dataset.keys())[0]])
	num_feature = len(dataset.keys())

	split = {}
	split_val = {}
	for feature in dataset.keys():
		split[feature] = []

		if type(dataset[feature][0]) == float : #numerical feature
			maks = dataset[feature][0]
			mins = dataset[feature][0]
			for data in dataset[feature]:
				maks = max(maks, data)
				mins = min(mins, data)

			cut_point = (maks-mins)*random.uniform(0, 1) + mins

			split_val[feature] = cut_point
			for data in dataset[feature]:
				split[feature].append(data > cut_point)
		else :
			selected_sample = random.choice(dataset[feature])

			split_val[feature] = selected_sample
			for data in dataset[feature]:
				split[feature].append(data == selected_sample)

	return split, split_val


class _Node():
	def __init__(self):
		self.left_child = None
		self.right_child = None
		self.split_feature = None
		self.split_val = None
		self.is_leaf = False
		self.leaf_value = None
		self.score = 0
		self.depth = 0

class ExtraTreeClassifier():
	def __init__(self, max_features, min_split):
		self.MAX_DEPTH = 100
		self.max_features = max_features
		self.min_split = min_split
		self.root = _Node()

	def __train(self, data, label, depth):
		node = _Node()
		num_row = len(data[list(data.keys())[0]])
		num_feature = len(data.keys())

		if ( (num_row < self.min_split) or (len(set(label)) <= 1) ):
			node.is_leaf = True
			node.leaf_value = _max_freq(label)
			node.depth = depth

			return node

		selected_feature = list(data.keys())
		random.shuffle(selected_feature)

		sub_data = {}

		for idx in range(min(len(selected_feature), self.max_features)):
			sub_data[selected_feature[idx]] = data[selected_feature[idx]]

		split, split_val = _random_split(sub_data)
		split_score = _score(label, split)

		max_info_gain_feature = selected_feature[0]
		max_info_gain_score = split_score[selected_feature[0]]
		for feature in selected_feature:
			if max_info_gain_score < split_score[feature]:
				max_info_gain_feature = feature
				max_info_gain_score = split_score[feature]

		max_info_gain_val = split_val[feature]

		if ( (len(set(split[max_info_gain_feature])) == 1) or (depth > self.MAX_DEPTH) ):
			node.is_leaf = True
			node.leaf_value = _max_freq(label)
			node.depth = depth

			return node

		else :
			data_left_child =  {}
			data_right_child =  {}
			label_left_child = []
			label_right_child = []

			for key in split.keys():
				data_left_child[key] = []
				data_right_child[key] = []

			for idx in range(len(split[max_info_gain_feature])):
				val = split[max_info_gain_feature][idx]
				if val == True:
					label_left_child.append(label[idx])
					for key in split.keys():
						data_left_child[key].append(data[key][idx])
				else :
					label_right_child.append(label[idx])
					for key in split.keys():
						data_right_child[key].append(data[key][idx])

			node.left_child = self.__train(data_left_child, label_left_child, depth+1)
			node.right_child = self.__train(data_right_child, label_right_child, depth+1)
			node.split_feature = max_info_gain_feature
			node.split_val = split_val[node.split_feature]
			node.score = max_info_gain_val
			node.depth = depth

			return node

	def __predict(self, node, datapoint):
		if node.is_leaf == True:
			return node.leaf_value

		if type(datapoint[node.split_feature]) == float:
			if datapoint[node.split_feature] > node.split_val:
				return self.__predict(node.left_child, datapoint)
			else:
				return self.__predict(node.right_child, datapoint)
		else:
			if datapoint[node.split_feature] == node.split_val:
				return self.__predict(node.left_child, datapoint)
			else:
				return self.__predict(node.right_child, datapoint)

	def train(self, data, label):
		self.root = self.__train(data, label, 1)
		return self.predict(data)

	def predict(self, data):
		label = []
		for idx in range(len(data[list(data.keys())[0]])):
			datapoint = {}

			for feature in data.keys():
				datapoint[feature] = data[feature][idx]

			label.append(self.__predict(self.root, datapoint))

		return label

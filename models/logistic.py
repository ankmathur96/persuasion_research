from sklearn import linear_model
import numpy as np

DATA_DIR = '../brat-v1.3_Crunchy_Frog/data/speeches/'
labels = ['Name_Calling','Glittering_Generalities','Testimonial','Plain_Folks','Credit_Claiming','Stereotyping','Slogans','Humor','Warmth','Patriotism','Repetition','Fear','Emotional_Anecdotes','Bandwagon', 'Transfer']
def load_data():
	dataset = []
	with open(DATA_DIR + 'featurized_bw.csv', 'r') as data_f:
		for i, line in enumerate(data_f):
			if i == 0:
				continue
			tokens = line.split(',')
			data_vector = np.array([int(x) for x in tokens])
			dataset.append(data_vector)
	with open(DATA_DIR + 'labels_bw.csv', 'r') as in_f:
		labels = []
		for line in in_f:
			label = int(line)
			labels.append(label)
	return np.array(labels), np.vstack(dataset)

def merge_all_lists_but_one(lists, ignore):
	merged = []
	for i in range(len(lists)):
		if i == ignore:
			continue
		merged.append(lists[i])
	return np.vstack(merged)

def k_fold_validation(X_train, labels_train, k):
	divisions = np.linspace(0, len(X_train), k+1, dtype=np.int64)
	all_train, all_labels = [], []
	for i in range(len(divisions)-1):
		all_train.append(X_train[divisions[i]:divisions[i+1]])
		labels_division = labels_train[divisions[i]:divisions[i+1]]
		all_labels.append(np.reshape(labels_division, (len(labels_division), 1)))
	all_divisions = []
	for j in range(k):
		all_divisions.append((merge_all_lists_but_one(all_train, j), merge_all_lists_but_one(all_labels, j), (all_train[j], all_labels[j])))
	return all_divisions

def confusion_matrix(X_train, labels_train, pred_train):
	print(pred_train)
	c_mat = np.zeros((len(labels), len(labels)))
	incorrect = 0
	for i in range(pred_train.shape[0]):
		correct_label, pred_label = labels_train[i], pred_train[i]
		if correct_label != pred_label:
			incorrect += 1

		c_mat[correct_label][pred_label] += 1
	# c_mat = c_mat / len(pred_train)
	print(str(incorrect))
	return c_mat

def pretty_print_matrix(mat, column_labels, row_labels):
	# shorten the labels
	column_labels = ["".join([x[0] for x in label.split('_')]) for label in column_labels]
	row_labels = ["".join([x[0] for x in label.split('_')]) for label in row_labels]
	label_space = '    '
	print('      ' + '     '.join(column_labels))
	for i in range(mat.shape[0]):
		row_to_print = [row_labels[i]] + ['{0:.2f}'.format(x) for x in mat[i]]
		print(label_space.join(row_to_print))

def check_labels_mat(labels_mat):
	unique_labels = set()
	for label in labels_mat:
		unique_labels.add(label)
	if len(unique_labels) <= 1:
		return False
	return True

regr = linear_model.LogisticRegression(C=1e-5)

labels_train, X_train = load_data()
print(len(X_train))
k_fold_divisions = k_fold_validation(X_train, labels_train, 5)
label_occurences = {}
for label in labels_train:
	if label in label_occurences:
		label_occurences[label] += 1
	else:
		label_occurences[label] = 1
print(label_occurences)
for k in label_occurences:
	label_occurences[k] /= len(labels_train)
# confirm that labels aren't super prevalent
summed = 0
for label in label_occurences:
	summed += label_occurences[label]
print(label_occurences, summed)
# logistic regression k-class model.
all_labels_logistic_model = regr.fit(X_train, labels_train)
print("Training accuracy for k-class classifier: " + str(all_labels_logistic_model.score(X_train, labels_train)))
# print the confusion matrix for X_train:
training_accuracies, validation_accuracies = [], []
for division in k_fold_divisions:
	regr = linear_model.LogisticRegression(C=1e-5)
	X_div, labels_div = division[0], division[1]
	validation_train, validation_labels = division[2]
	all_labels_logistic_model = regr.fit(X_div, labels_div)
	training_accuracies.append(all_labels_logistic_model.score(X_div, labels_div))
	validation_accuracies.append(all_labels_logistic_model.score(validation_train, validation_labels))
print(validation_accuracies)
print("Training accuracy for k-class classifier: " + str(sum(training_accuracies) / len(training_accuracies)))
print("Validation accuracy for k-class classifier: " + str(sum(validation_accuracies) / len(validation_accuracies)))
# confusion_matrix = confusion_matrix(X_train, labels_train, all_labels_logistic_model.predict(X_train))
# pretty_print_matrix(confusion_matrix, labels, labels)
# # try for just a specific category:

# for classified in range(len(labels)):
# 	accuracies = []
# 	for division in k_fold_divisions:
# 		X_train, labels_train = division[0], division[1]
# 		labels_mat = [x[0] for x in labels_train]
# 		for i, label in enumerate(labels_train):
# 			if label != classified:
# 				labels_mat[i] = classified + 1
# 		# validate the labels
# 		if not check_labels_mat(labels_mat):
# 			continue
# 		one_label_logistic_model = regr.fit(X_train, labels_mat)
# 		# print(one_label_logistic_model.predict(X_train))
# 		accuracy = one_label_logistic_model.score(X_train, labels_mat)
# 		accuracies.append(float(accuracy))
# 	if len(accuracies) == 0:
# 		continue
# 	print("Training accuracy for label " + str(labels[classified]) + ": " + str(np.average(accuracies)))



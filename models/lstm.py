import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import TimeDistributedDense, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
EMBEDDED_LAYER_SPACE = 32
DATA_DIR = '../brat-v1.3_Crunchy_Frog/data/speeches/'
labels = ['Name_Calling','Glittering_Generalities','Testimonial','Plain_Folks','Credit_Claiming','Stereotyping','Slogans','Humor','Warmth','Patriotism','Repetition','Fear','Emotional_Anecdotes','Bandwagon', 'Transfer']
def load_data():
	dataset = []
	with open(DATA_DIR + 'featurized_sv.csv', 'r') as data_f:
		for i, line in enumerate(data_f):
			if i == 0:
				tokens = line.split(',')
				word_space_size = int(tokens[0])
				max_sentence_length = int(tokens[1])
				continue
			tokens = line.split(',')
			data_vector = np.array([int(x) for x in tokens])
			dataset.append(data_vector)
	with open(DATA_DIR + 'labels_sv.csv', 'r') as in_f:
		labels = []
		for line in in_f:
			label = int(line)
			labels.append(label)
	return np.array(labels), np.vstack(dataset), word_space_size, max_sentence_length

def check_labels_mat(labels_mat):
	unique_labels = set()
	for label in labels_mat:
		unique_labels.add(label)
	if len(unique_labels) <= 1:
		return False
	return True

# RESEARCH Dataset
DATA_FILES = 'all_data/'
# labels = ['Name_Calling']
text_labels = ['Name_Calling','Glittering_Generalities','Testimonial','Plain_Folks','Credit_Claiming','Stereotyping','Slogans','Humor','Warmth','Patriotism','Repetition','Fear','Emotional_Anecdotes','Bandwagon', 'Transfer']
USELESS = {" ", "-", "–", '"', "'", ";", ":", ".", "!", "?", ",", "--"}
labels_index = {a : i for i, a in enumerate(text_labels)}
# annotated text
def clean_sentence(sentence):
    cleaned = []
    for word in sentence:
        if word in USELESS:
            continue
        while word[-1] in USELESS or word == '':
            word = word[:-1]
        cleaned.append(word)
    return cleaned

texts = []
labels = []
print('Loading annotations and full dataset from files')
for z, file in enumerate(os.listdir(DATA_FILES)):
    sys.stdout.write("Dataset File: %d   \r" % (z))
    sys.stdout.flush()
    if file.split('.')[1] == 'ann':
        with open(DATA_FILES + file, 'r') as f:
            for annotation_line in f:
                tokens = annotation_line.split()
                label = tokens[1]
                if label not in labels_index:
                    continue
                l_range, r_range = tokens[2], tokens[3]
                sentence = tokens[4:]
                sentence = clean_sentence(sentence)
                string_sentence = ' '.join(sentence)
                texts.append(string_sentence)
                labels.append(labels_index[label])
print('Found %s texts.' % len(texts))
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

from keras.utils import to_categorical
import numpy as np

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

from keras.layers import Embedding, Input, Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import Model
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print(labels_train.shape)
lstm_network = Sequential()
lstm_network.add(embedding_layer)
lstm_network.add(LSTM(100, return_sequences=True))
lstm_network.add(TimeDistributedDense(15))
lstm_network.add(Activation('softmax'))
lstm_network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# lstm_network.summary()
print "Training accuracy:"
lstm_network.fit(X_train, labels_train, nb_epoch=3, batch_size=64)
scores = lstm_network.evaluate(X_train, labels_mat, verbose=0)
print(scores)
# for classified in range(len(labels)):
# 	labels_mat = np.copy(labels_train)
# 	for i, label in enumerate(labels_train):
# 		if label != classified:
# 			labels_mat[i] = classified + 1
# 	# validate the labels
# 	if not check_labels_mat(labels_mat):
# 		continue
# 	lstm_network = Sequential()
# 	lstm_network.add(Embedding(word_space_size, EMBEDDED_LAYER_SPACE, input_length=max_sentence_length))
# 	lstm_network.add(LSTM(100))
# 	lstm_network.add(Dense(15, activation='softmax'))
# 	lstm_network.compile(metrics=['accuracy'])
# 	# lstm_network.summary()
# 	print "Training accuracy for label " + str(labels[classified]) + ":"
# 	lstm_network.fit(X_train, labels_mat, nb_epoch=3, batch_size=64)
# 	scores = lstm_network.evaluate(X_train, labels_mat, verbose=0)
# 	print(scores)
# model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
# model.add(MaxPooling1D(pool_length=2))

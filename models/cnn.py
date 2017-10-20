import os
import sys
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
VALIDATION_SPLIT = 0.05
EMBEDDING_DIM = 100
N_EPOCHS = 100
TEXT_DATA_DIR = '20_newsgroup'
GLOVE_DIR = 'glove.6B'
# texts = []  # list of text samples
# labels_index = {}  # dictionary mapping label name to numeric id
# labels = []  # list of label ids
# for name in sorted(os.listdir(TEXT_DATA_DIR)):
#     path = os.path.join(TEXT_DATA_DIR, name)
#     if os.path.isdir(path):
#         label_id = len(labels_index)
#         labels_index[name] = label_id
#         for fname in sorted(os.listdir(path)):
#             if fname.isdigit():
#                 fpath = os.path.join(path, fname)
#                 f = open(fpath, encoding='latin-1')
#                 t = f.read()
#                 i = t.find('\n\n')  # skip header
#                 if i > 0:
#                     t = t[i:]
#                 texts.append(t)
#                 f.close()
#                 labels.append(label_id)

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

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=N_EPOCHS, batch_size=128)

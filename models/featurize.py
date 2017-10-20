import os
import sys
import csv
import numpy as np
DATA_FILES = '../brat-v1.3_Crunchy_Frog/data/speeches/'
labels = ['Name_Calling','Glittering_Generalities','Testimonial','Plain_Folks','Credit_Claiming','Stereotyping','Slogans','Humor','Warmth','Patriotism','Repetition','Fear','Emotional_Anecdotes','Bandwagon', 'Transfer']
USELESS = {" ", "-", '"', "'", ";", ":", ".", "!", "?", ",", "--"}
MAX_SENTENCE_LENGTH = 200
def clean_sentence(sentence):
    cleaned = []
    for word in sentence:
        if word in USELESS:
            continue
        while word[-1] in USELESS or word == '':
            word = word[:-1]
        cleaned.append(word)
    return cleaned

def featurize(dataset_dir):
    # compute the space of words
    all_words = {}
    full_corpus = []
    for file in os.listdir(dataset_dir):
        if file.split('.')[1] == 'ann':
            with open(dataset_dir + file, 'r') as f:
                for annotation_line in f:
                    tokens = annotation_line.split()
                    label = tokens[1]
                    if label not in labels:
                        continue
                    l_range, r_range = tokens[2], tokens[3]
                    sentence = tokens[4:]
                    sentence = clean_sentence(sentence)
                    full_corpus.append((sentence, label))
                    string_sentence = " ".join(sentence)
                    for word in sentence:
                        if word in all_words:
                            all_words[word] += 1
                        else:
                            all_words[word] = 1
    # create a bag of words representation of the dataset for logistic
    word_list = list(all_words.keys())
    with open(dataset_dir + 'featurized_bw.csv', 'w') as data_f:
        data_writer = csv.writer(data_f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow(word_list)
        with open(dataset_dir + 'labels_bw.csv', 'w') as labels_out_f:
            labels_writer = csv.writer(labels_out_f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for example in full_corpus:
                sentence, label = example[0], example[1]
                featurized_sentence = np.zeros(len(word_list))
                for word in sentence:
                    featurized_sentence[word_list.index(word)] += 1
                data_writer.writerow(featurized_sentence.astype(int))
                labels_writer.writerow([labels.index(label)])
    # create a sequential representation of words to represent a sentence - pad to a max length.
    # sorted from most frequent to least frequent
    sorted_word_list = [x[0] for x in sorted(all_words.items(), key=lambda x: x[1])[::-1]]
    with open(dataset_dir + 'featurized_sv.csv', 'w') as data_f:
        data_writer = csv.writer(data_f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow([len(sorted_word_list), MAX_SENTENCE_LENGTH])
        with open(dataset_dir + 'labels_sv.csv', 'w') as labels_out_f:
            labels_writer = csv.writer(labels_out_f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for example in full_corpus:
                sentence, label = example[0], example[1]
                featurized_sentence = np.zeros(MAX_SENTENCE_LENGTH)
                for i, word in enumerate(sentence):
                    featurized_sentence[i] = sorted_word_list.index(word)
                data_writer.writerow(featurized_sentence.astype(int))
                labels_writer.writerow([labels.index(label)])
if len(sys.argv) > 1:
    DATA_FILES = sys.argv[1]
featurize(DATA_FILES)


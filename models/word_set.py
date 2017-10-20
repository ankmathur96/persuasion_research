import sys
import os
import nltk
import pickle
import numpy as np
import heapq
all_words = {}
full_corpus = []
full_dataset = []
DATA_FILES = 'all_data/'
SPEECH_CORPUS = '../fetch_data/output/'
TOP_K = 20
EXPANSION_DISTANCE = 2
A_1 = 0.8
A_2 = 0.2
# labels = ['Name_Calling']
labels = ['Name_Calling','Glittering_Generalities','Testimonial','Plain_Folks','Credit_Claiming','Stereotyping','Slogans','Humor','Warmth','Patriotism','Repetition','Fear','Emotional_Anecdotes','Bandwagon', 'Transfer']
IRRElEVANT_WORDS = {'and', 'or', 'if', 'the', 'a', 'an', 'of', 'they', 'are', 'this', 'that', 'hillary', 'donald', 'she', 'is', 'to', 'in', 'was', 'my', 'we', 'who', 'so', 'i', 'there', 'on'}
USELESS = {" ", "-", "–", '"', "'", ";", ":", ".", "!", "?", ",", "--"}
banned_pos = {'IN', 'CC', 'PRP$', 'PRP'}

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
def normalize_clusters(category_clusters):
    for c in labels:
        category_set = category_clusters[c]
        total = 0
        for k in category_set:
            total += category_set[k]
        for k in category_set:
            category_set[k] /= total

if os.path.isfile('full_corpus.pk') and os.path.isfile('full_dataset.pk'):
    print('Loading annotations and full dataset from cache')
    with open('full_corpus.pk', 'rb') as in_f:
        full_corpus = pickle.load(in_f)
    with open('full_dataset.pk', 'rb') as in_f:
        full_dataset = pickle.load(in_f)
else:
    print('Loading annotations and full dataset from files')
    for z, file in enumerate(os.listdir(DATA_FILES)):
        sys.stdout.write("Dataset File: %d   \r" % (z))
        sys.stdout.flush()
        if file.split('.')[1] == 'ann':
            with open(DATA_FILES + file, 'r') as f:
                for annotation_line in f:
                    tokens = annotation_line.split()
                    label = tokens[1]
                    if label not in labels:
                        continue
                    l_range, r_range = tokens[2], tokens[3]
                    sentence = tokens[4:]
                    sentence = clean_sentence(sentence)
                    string_sentence = ' '.join(sentence)
                    full_corpus.append((sentence, label, string_sentence))
        elif file.split('.')[1] == 'txt':
            with open(DATA_FILES + file, 'r') as f:
                text = f.read()
                sentences = nltk.sent_tokenize(text)
                full_dataset.extend(sentences)
    print()
    print('----'*10)
    print('Saving parsed files as cache.')
    with open('full_corpus.pk', 'wb') as out_f:
        pickle.dump(full_corpus, out_f)
    with open('full_dataset.pk', 'wb') as out_f:
        pickle.dump(full_dataset, out_f)
print('Beginning analytics portion of code.')
all_words = set()
category_core_clusters = {}
if '--cached' in sys.argv:
    print('Using cached clusters')
    with open('core_clusters.pk', 'rb') as i_f:
        category_core_clusters = pickle.load(i_f)
    with open('expanded_clusters.pk', 'rb') as i_f2:
        category_expanded_clusters = pickle.load(i_f2)
    with open('all_words.pk', 'rb') as i_f3:
        all_words = pickle.load(i_f3)
else:
    print('Generating core clusters')
    for c in labels:
        word_set = {}
        for s in full_corpus:
            if s[1] == c:
                # print('instance', s[0])
                tokenized = nltk.word_tokenize(s[2])
                # print(nltk.pos_tag(tokenized))
                cleaned_sentence = [w[0] for w in nltk.pos_tag(tokenized) if w[1] not in banned_pos]
                for w in cleaned_sentence:
                    w = w.lower()
                    if w.lower() in IRRElEVANT_WORDS or w in USELESS:
                        continue
                    else:
                        if w in word_set:
                            word_set[w] += 1
                        else:
                            word_set[w] = 1
                        all_words.add(w)
        word_list = sorted([(k,v) for k,v in word_set.items()], key=lambda x: x[1])[::-1][:TOP_K]
        cropped_word_set = {x[0]:x[1] for x in word_list}
        category_core_clusters[c] = cropped_word_set
    # # all text
    speech_dataset = []
    ignored = 0
    for file in os.listdir(SPEECH_CORPUS):
        with open(SPEECH_CORPUS + file, 'r') as f:
            try:
                contents = f.read()
                sentences = nltk.sent_tokenize(contents)
                speech_dataset.extend(sentences)
            except UnicodeDecodeError:
                ignored += 1
    print('filled speech dataset with ' + str(len(os.listdir(SPEECH_CORPUS)) - ignored) + ' speeches')
    category_expanded_clusters = {}
    for c in labels:
        category_expanded_clusters[c] = {}
    print(str(len(speech_dataset)) + ' total sentences')
    for j,s in enumerate(speech_dataset):
        sys.stdout.write("Sentence: %d   \r" % (j))
        sys.stdout.flush()
        tokenized = nltk.word_tokenize(s)
        cleaned_sentence = [w[0] for w in nltk.pos_tag(tokenized) if w[1] not in banned_pos]
        for c in labels:
            for i, w in enumerate(cleaned_sentence):
                w = w.lower()
                if w.lower() in IRRElEVANT_WORDS or w in USELESS:
                    continue
                if w in category_core_clusters[c]:
                    slice_left = i - EXPANSION_DISTANCE if i - EXPANSION_DISTANCE > 0 else 0
                    slice_right = i + EXPANSION_DISTANCE # no negative impact from slicing past the end
                    promoted_words = cleaned_sentence[slice_left:slice_right+1]
                    for expanded_w in promoted_words:
                        if expanded_w == w:
                            continue
                        if expanded_w in category_expanded_clusters[c]:
                            category_expanded_clusters[c][expanded_w] += 1
                        else:
                            category_expanded_clusters[c][expanded_w] = 1
                        all_words.add(expanded_w)
    # normalize clusters
    normalize_clusters(category_core_clusters)
    normalize_clusters(category_expanded_clusters)
    if '--make-cache' in sys.argv:
        print('Generating cache')
        with open('core_clusters.pk', 'wb') as o_f:
            pickle.dump(category_core_clusters, o_f)
        with open('expanded_clusters.pk', 'wb') as o_f2:
            pickle.dump(category_expanded_clusters, o_f2)
        with open('all_words.pk', 'wb') as o_f3:
            pickle.dump(all_words, o_f3)
# print(category_core_clusters)
# print(category_expanded_clusters)
# compute P(c | w) for all words and all categories.
word_probs = {}
for w in all_words:
    word_vec = np.zeros(len(labels))
    for i,c in enumerate(labels):
        if w in category_core_clusters[c]:
            word_vec[i] += A_1 * category_core_clusters[c][w]
        if w in category_expanded_clusters[c]:
            word_vec[i] += A_2 * category_expanded_clusters[c][w]
    word_probs[w] = word_vec

def classify(category_core_clusters, category_expanded_clusters, s):
    categorization = np.array(len(labels))
    tokenized = nltk.word_tokenize(s)
    cleaned_sentence = [w[0] for w in nltk.pos_tag(tokenized) if w[1] not in banned_pos]
    total_vec = np.zeros(len(labels))
    for w in cleaned_sentence:
        w = w.lower()
        if w.lower() in IRRElEVANT_WORDS or w in USELESS:
            continue
        # if w not in word_probs:
        #     continue
        total_vec += word_probs[w]
    total_vec /= len(cleaned_sentence)
    return total_vec

def classify_dataset(corpus, category_core_clusters, category_expanded_clusters):
    correct = 0.0
    for sentence in corpus:
        result = classify(category_core_clusters, category_expanded_clusters, sentence[2])
        label_index = labels.index(sentence[1])
        top_3 = heapq.nlargest(3, range(len(result)), result.take)
        if label_index in top_3:
            correct += 1
    accuracy = correct / len(corpus)
    print('Training accuracy:', str(accuracy), '; n: ', str(len(corpus)))


    # for every annotation, see if a k-gram of it is in the other annotation list.
    # if an annotation exists with the same label, add an agreement. else, a disagreement.
classify_dataset(full_corpus, category_core_clusters, category_expanded_clusters)

# print('Patriotism')
# print(category_core_clusters['Patriotism'])
# print('Fear')
# print(category_core_clusters['Fear'])
# classification_vector = classify(category_core_clusters, category_expanded_clusters, 'America is the greatest country in the world')
# for i, n in enumerate(classification_vector):
#     if n > 0.01:
#         print(labels[i])
# print(classification_vector)
# print(classification_vector[labels.index('Patriotism')])

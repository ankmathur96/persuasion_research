import sys
import os
import nltk
import pickle
import numpy as np
import heapq

all_words = {}
full_corpus = []
full_dataset = []
DATA_FILES = '../brat-v1.3_Crunchy_Frog/data/speeches/'
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

def compute_overlap(l_range, r_range, l_range2, r_range2):
    if l_range <= l_range2 and l_range2 <= r_range:
        # overlap
        return min(r_range, r_range2) - l_range2
    elif l_range2 <= l_range and l_range <= r_range2:
        return min(r_range, r_range2) - l_range
    else:
        return 0

# compute cohen's kappa
def compute_similarity_score(ann_list_1, ann_list_2, category_percentages):
    k = 3
    n_agreements, n_disagreements = 0, 0
    corrections = []
    for ann in ann_list_1:
        l_s, label, str_s, f_n, l_range, r_range = ann
        agreement = False
        for ann_2 in ann_list_2:
            l_s2, label2, str_s2, f_n2, l_range2, r_range2 = ann_2
            if f_n == f_n2 and compute_overlap(l_range, r_range, l_range2, r_range2) != 0:
                # overlap.
                if label == label2:
                    # we can check here for percentage based chance of random agreement.
                    corrections.append(category_percentages[label])
                    agreement = True
                    break
        if agreement:
            n_agreements += 1
        else:
            n_disagreements += 1
    total = n_agreements + n_disagreements
    observed_agreement = n_agreements / total
    random_agreement = sum(corrections) / len(corrections)
    print('Observed Agreement:', observed_agreement)
    print('Random Agreement:', random_agreement)
    print('Annotator 1:', len(ann_list_1))
    print('Annotator 2:', len(ann_list_2))
    return (observed_agreement - random_agreement) / (1 - random_agreement)
category_percentages = [{label : 0 for label in labels}, 0]
def read_ann_list(path, ref_path):
    annotation_list = []
    reference_files = os.listdir(ref_path)
    for file in os.listdir(path):
        if file.split('.')[1] == 'ann':
            ref_file_name = file.split('.')[0] + '.txt'
            if ref_file_name not in reference_files:
                print('Reference file', file.split('.')[0] + '.txt', 'not found.')
            with open(path + file, 'r') as f:
                for annotation_line in f:
                    if annotation_line == '' or annotation_line == '\n':
                        continue
                    tokens = annotation_line.split()
                    label = tokens[1]
                    if label not in labels:
                        continue
                    l_range, r_range = tokens[2], tokens[3]
                    sentence = tokens[4:]
                    sentence = clean_sentence(sentence)
                    string_sentence = ' '.join(sentence)
                    try:
                        annotation_list.append((sentence, label, string_sentence, ref_file_name, int(l_range), int(r_range)))
                        category_percentages[0][label] += 1
                        category_percentages[1] += 1
                    except ValueError:
                        continue
    return annotation_list


ann_list = [('America is the greatest country', 'Patriotism'), ('I loved my mother more than anyone else', 'Emotional_Anecdotes')]
ann_list_2 = [('America is the greatest country', 'Patriotism'), ('I loved my mother more', 'Plain_Folks')]

ana_anns = read_ann_list('ana_anns/', 'shared_texts/')
ankit_anns = read_ann_list('fake_anns/', 'shared_texts/')
category_percentages = {k : v / category_percentages[1] for k,v in category_percentages[0].items()}
print('Category Priors:', category_percentages)
print(compute_similarity_score(ana_anns, ankit_anns, category_percentages))

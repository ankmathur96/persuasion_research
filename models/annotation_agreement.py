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

# get the ranges for the annotations for a given document.
def get_ann_ranges(ann_list, document_char_size):
    # create a list of annotation barriers of form
    # (index, type [start or end], label)
    markers = []
    for a in ann_list:
        markers.append((a[4], 'start', a[1]))
        markers.append((a[5], 'end', a[1]))
    # sort by the barrier index
    sorted_ann_markers = sorted(markers, lambda x: x[0])
    final_ann_list = []
    active_label_set = {}
    i = 0
    ann_idx = 0
    marker_idx = 0
    start_of_current_interval = 0
    while i < document_char_size:
        if i == markers[marker_idx][0]:
            # we've hit a marker.
            # [x, y) - range specification.
            final_ann_list.append((start_of_current_interval, i, active_label_set if len(active_label_set) > 0 else {'unlabeled'}))
            start_of_current_interval = i
            if markers[marker_idx][1] == 'start':
                # add the label to the marked set.
                active_label_set.add(markers[marker_idx][2])
            elif markers[marker_idx][1] == 'end':
                active_label_set.remove(markers[marker_idx][2])
            marker_idx += 1
        i += 1
    return final_ann_list

def per_word_ann_list(ann_file_path, doc_path, document_name):
    annotations = get_anns_for_document(ann_file_path, document_name)
    with open(doc_path + document_name, 'r') as in_f:
        d = in_f.read()
        doc_length = len(d)
        ann_ranges = get_ann_ranges(annotations, doc_length)
        ann_ranges_w_text = [x.append(d[x[0]:x[1]) for x in ann_ranges]

        return annotation_list

def get_all_annotations(path, doc_path):
    annotation_list = []
    reference_files = os.listdir(doc_path)
    for file in os.listdir(path):
        if file.split('.')[1] == 'ann':
            ref_file_name = file.split('.')[0] + '.txt'
            if ref_file_name not in reference_files:
                raise NameError('Reference file', file.split('.')[0] + '.txt', 'not found.')
            annotation_list.extend(get_anns_for_document(path + file, ref_file_name))
    return annotation_list

def get_anns_for_document(ann_path, ref_file_name):
    annotation_list = []
    with open(ann_path, 'r') as f:
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



ann_list = [('America is the greatest country', 'Patriotism'), ('I loved my mother more than anyone else', 'Emotional_Anecdotes')]
ann_list_2 = [('America is the greatest country', 'Patriotism'), ('I loved my mother more', 'Plain_Folks')]

ana_anns = get_all_annotations('ana_anns/', 'shared_texts/')
ankit_anns = get_all_annotations('fake_anns/', 'shared_texts/')
category_percentages = {k : v / category_percentages[1] for k,v in category_percentages[0].items()}
print('Category Priors:', category_percentages)
print(compute_similarity_score(ana_anns, ankit_anns, category_percentages))

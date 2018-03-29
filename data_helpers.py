import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def fit_transform(x_test, max_document_length):
    for x in x_test:
        while len(x) < max_document_length :
            add = [0 for i in range(6)]
            x.append(add)
    return x_test

def load_data_and_labels(data_file):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    y, x_text =[], []
    examples = open(data_file, "r", encoding='utf-8').readlines()
    #examples = [s.strip() for s in examples]
    # Split by words
    for k in examples :
        y_data = k.split(',')[-1].strip()
        y.append(y_data)
        k_raw = [list(map(int, i.split())) for i in k.split(',')[:-1]]
        x_text.append(k_raw)
    category = list(set(y))
    category.sort()
    return [x_text, y, category]


def transform_labels(label, category):
    #Generate labels
    labels = []
    for i in range(len(label)):
        element = [0 for x in range(len(category))]
        element[category.index(label[i])] = 1
        labels.append(element)
    return labels

def eval_data(data_file):
    examples = np.array(open(data_file, "r", encoding='utf-8').readlines())
    shuffle_indices = np.random.permutation(np.arange(len(examples)))
    data = examples[shuffle_indices]
    eval_sample_index = -1 * int(0.1 * float(len(data)))
    data = data[:eval_sample_index]
    open('./data/test.txt', "w", encoding='utf-8').writelines(data)



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    x_text, y, category = load_data_and_labels('./data/cmm_paths.txt')
    y =transform_labels(y,category)
    max_document_length = max([len(x) for x in x_text])
    print(max_document_length)
    for k in y:
        print(k)




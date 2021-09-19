import torch
import sys
sys.path.append("..")
from antirouge.config import *
import os
import pickle
import json
from antirouge.main import *

def write_file(articles, summaries, labels, shuffle, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    ct = 0
    for index in shuffle:
        ct += 1
        if ct % 1000 == 0:
            print(ct)
        file = os.path.join(folder, str(index) + ".json")
        with open(file, "w", encoding='utf-8') as f:
            sample = [{'article': articles[index][0], 'summary': summaries[index][0], 'label': float(labels[index][0])},
            {'article': articles[index][1], 'summary': summaries[index][1], 'label': float(labels[index][1])}]
            json.dump(sample, f)

def save_data(output_path, fake_method, fake_extra_option=None):
    print('loading data ..')
    (articles, reference_summaries, reference_labels, fake_summaries,
     fake_labels, keys) = load_data_helper(fake_method,
                                           'USE',
                                           30000,
                                           1,
                                           fake_extra_option)
    print('merging data ..')
    articles, summaries, labels = merge_summaries(articles,
                                                  reference_summaries,
                                                  reference_labels,
                                                  fake_summaries,
                                                  fake_labels)
    group = 2

    articles = np.array(np.split(articles, len(articles) / group))
    summaries = np.array(np.split(summaries, len(summaries) / group))
    labels = np.array(np.split(labels, len(labels) / group))

    with open(SHUFFLE_FILE, "rb") as f:
        indices = pickle.load(f)
        assert(indices.shape[0] == articles.shape[0])
        print("Load saved shuffled index.")

    num_validation_samples = int(0.1 * articles.shape[0])
    train = indices[:-num_validation_samples*2]
    val = indices[-num_validation_samples*2:-num_validation_samples]
    test = indices[-num_validation_samples:]

    train_folder = os.path.join(output_path, fake_method, "" if fake_extra_option == None else fake_extra_option, "train")
    val_folder = os.path.join(output_path, fake_method, "" if fake_extra_option == None else fake_extra_option, "val")
    test_folder = os.path.join(output_path, fake_method, "" if fake_extra_option == None else fake_extra_option, "test")

    article_length = np.hstack([[articles[index][i].shape[0] for index in train] for i in range(2)])
    summary_length = np.hstack([[summaries[index][i].shape[0] for index in train] for i in range(2)])
    
    article_length = np.sort(article_length)
    summary_length = np.sort(summary_length)

    print((article_length, summary_length))

    '''
    print("Train...")
    write_file(articles, summaries, labels, train, train_folder)
    print("Validation...")
    write_file(articles, summaries, labels, val, val_folder)
    print("Test...")
    write_file(articles, summaries, labels, test, test_folder)
    '''
        
        
    

if __name__ == '__main__':

    #input_path = os.path.join(DATA_DIR, "cnn\\stories")
    output_path = os.path.join(DATA_DIR, "samples")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    save_data(output_path, 'neg')
    save_data(output_path, 'mutate', 'add')
    save_data(output_path, 'mutate', 'delete')
    save_data(output_path, 'mutate', 'replace')

    '''
    print(input_path)
    ct = 0
    for key in keys:
        ct += 1
        if ct % 100 == 0:
            print(ct)
        os.system('copy ' + os.path.join(input_path, key) + ' ' + os.path.join(output_path, key)) 
    '''
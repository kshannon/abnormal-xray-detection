#!/usr/bin/env python

# Script to take the label column from train_labeled_studies and populate it
# into the train_image_paths csv (for both train and validate)
# Test script by uncommenting #DEBUG lines and commenting out preceeding lines.

import sys
import os
from configparser import ConfigParser
import pandas as pd
import numpy as np
import re


def generate_labels(data):
    '''crawl through csv, regex for label in str path, and return list of labels'''
    labels = []
    for idx,row in data.iterrows():
        match = re.search(r'\w(positive)',row[-1]) #regex for str 'positive'
        if match:
            labels.append('1')
        else:
            labels.append('0')
    return labels

def create_dfs(path, header=None):
    return pd.read_csv(path, header=None, names=['path'])


def main():
    #### ---- ConfigParse Utility ---- ####
    config = ConfigParser()
    config.read('../../config/data_path.ini')

    try:
        csv_path = config.get('csv', 'csv_path') #grab .csv paths from .ini
        train_image_paths = config.get('csv', 'train_image_paths')
        valid_image_paths = config.get('csv', 'valid_image_paths')
        #DEBUG test on sample csvs
        # train_image_paths = config.get('sample', 'train_image_paths')
        # valid_image_paths = config.get('sample', 'valid_image_paths')
    except:
        print('could not read data_path.ini file. Try checking your paths.')
        sys.exit(1)

    # pandas to read and create csvs
    train = create_dfs(train_image_paths)
    valid = create_dfs(valid_image_paths)

    # regex to determine class label, turn list into series
    train_labels = pd.Series(generate_labels(train))
    valid_labels = pd.Series(generate_labels(valid))

    # add pd.Series() labels to the df
    train['labels'] = train_labels.values
    valid['labels'] = valid_labels.values

    # save df as csv, w/o header info
    train.to_csv(csv_path+'train.csv',header=False,index=False)
    valid.to_csv(csv_path+'valid.csv',header=False,index=False)
    #DEBUG test on sample data
    # train.to_csv(csv_path+'train_sample.csv',header=False,index=False)
    # valid.to_csv(csv_path+'valid_sample.csv',header=False,index=False)



if __name__ == '__main__':
    main()

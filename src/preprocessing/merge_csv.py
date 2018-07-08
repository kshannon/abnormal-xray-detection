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

#### ---- CSV File Names ---- ####
train_image_paths = 'train_image_paths.csv' #File provided by Stanford
valid_image_paths = 'valid_image_paths.csv' #File provided by Stanford

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
        sample_data = config.get('sample', 'sample_data') #build sample data
        complete_data = config.get('data', 'data_path') + 'MURA-v1.1/'
    except:
        print('could not read data_path.ini file. Try checking your paths.')
        sys.exit(1)

    for path in [sample_data, complete_data]:

        # pandas to read and create csvs
        train = create_dfs(path + train_image_paths)
        valid = create_dfs(path + valid_image_paths)

        # regex to determine class label, turn list into series
        train_labels = pd.Series(generate_labels(train))
        valid_labels = pd.Series(generate_labels(valid))

        # add pd.Series() labels to the df
        train['labels'] = train_labels.values
        valid['labels'] = valid_labels.values

        # save df as csv, w/o header info
        train.to_csv(path+'train.csv',header=False,index=False)
        valid.to_csv(path+'valid.csv',header=False,index=False)



if __name__ == '__main__':
    main()

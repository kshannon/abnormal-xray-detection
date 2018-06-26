#!/usr/bin/env python

# Script to take the label column from train_labeled_studies and populate it
# into the train_image_paths csv (for both train and validate)

# Kyle Shannon & Chris Chen
# Stanford MURA Challenge - 2018

import sys
import os
from configparser import ConfigParser
import pandas as pd
import numpy as np
import re


#### ---- ConfigParse Utility ---- ####
config = ConfigParser()
config.read('../../config/data_path.ini')

#### ---- Global Vars ---- ####
try:
    train_image_paths = config.get('sample', 'train_image_paths')
    valid_image_paths = config.get('sample', 'valid_image_paths')
    # train_image_paths = config.get('csv', 'train_image_paths')
    # valid_image_paths = config.get('csv', 'valid_image_paths')
except:
    print('could not read data_path.ini file. Make sure you paths are correct.')
    sys.exit(1)

def generate_labels(data):
    '''asd'''
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
    train = create_dfs(train_image_paths)
    valid = create_dfs(valid_image_paths)

    train_labels = generate_labels(train)
    valid_labels = generate_labels(valid)










if __name__ == '__main__':
    main()

#Function called by ‘train.py’
# Parses config files
# Constructs the Dataset(s), Model(s) and

import sys
from configparser import ConfigParser
import os
import numpy as np
import yaml
import tensorflow as tf
# Project Modules
from experiments import augmentation

'''requirements for yaml:
BATCH_SIZE
PREFETCH_SIZE
CHANNELS
CPU_CORES
MODEL = .py file maybe?
'''
# move these to config file.......
CHANNELS = 3
CPU_CORES = 2
BATCH_SIZE = 8
PREFETCH_SIZE = 1





#### ========= ConfigParse Utilities ========= ####
path_parser = ConfigParser()
path_parser.read('../../config/data_path.ini') #needs 2x ../ becaue train.py calls
try:
    data_path = path_parser.get('data', 'data_path')
    print("Data Path set to: " + data_path)
except:
    print('Error reading data_path.ini, try checking data paths in the .ini')
    sys.exit(1)


#### ========= Helper Functions ========= ####
def split_data_labels(csv_path, data_path):
    """ take CSVs with filepaths/labels and extracts them into parallel lists"""
    filenames = []
    labels = []
    with open(csv_path, 'r') as f:
        for line in f:
            new_line = line.strip().split(',')
            filenames.append(data_path + new_line[0])
            labels.append(float(new_line[1]))
    return filenames,labels

def preprocess_img(filename, label):
    """
    Read filepaths and decode into numerical tensors
    Process images --> tensorflow arrays
    """
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=CHANNELS) # Don't use tf.image.decode_image
    image = tf.image.convert_image_dtype(image, tf.float32) #convert to float values in [0, 1]
    return image, label

def create_tf_dataset(data, labels):
    """todo"""
    labels = tf.one_hot(tf.cast(labels, tf.uint8), 1) #cast labels to dim 2 tf obj
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(len(data)) #mioght not need this.... DEBUG
    dataset = dataset.repeat()
    dataset = dataset.map(preprocess_img, num_parallel_calls=CPU_CORES)
    dataset = dataset.map(augmentation.baseline, num_parallel_calls=CPU_CORES)
    dataset = dataset.batch(BATCH_SIZE) # (?, x, y) unknown batch size because the last batch will have fewer elements.
    dataset = dataset.prefetch(PREFETCH_SIZE) #single training step consumes n elements
    return dataset

def build_dataset(source_type,data_path=data_path):
    csv_path = data_path + 'MURA-v1.1/' + source_type
    filenames, labels = split_data_labels(csv_path, data_path)
    dataset = create_tf_dataset(filenames,labels)
    return dataset


#### ========= Build Model ========= ####
    ''' here we define a model... '''





# COMMENT_CHAR = '#'
# OPTION_CHAR =  '='
#
# def parse_config(filename):
#     options = {}
#     f = open(filename)
#     for line in f:
#         # First, remove comments:
#         if COMMENT_CHAR in line:
#             # split on comment char, keep only the part before
#             line, comment = line.split(COMMENT_CHAR, 1)
#         # Second, find lines with an option=value:
#         if OPTION_CHAR in line:
#             # split on option char:
#             option, value = line.split(OPTION_CHAR, 1)
#             # strip spaces:
#             option = option.strip()
#             value = value.strip()
#             # store in dictionary:
#             options[option] = value
#     f.close()
#     return options
#
# options = parse_config('config.ini')
# print options

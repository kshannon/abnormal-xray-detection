#!/usr/bin/env python

# Script that creates tensorflow data.Dataset object and pickles them
# Run this once to generate the pickle files. Make sure your paths are
# set correctly in the data_path.ini file. The paths should lead to your
# MURA .csv files


import sys
import os
import argparse
from configparser import ConfigParser
import pathlib
from glob import glob
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator

### # TODO: This script should become modular, will rip out the config parser
# and add that to train model, which will import this script, feed this script's
# functions the file paths etc etc TODO




### --- Helper Function --- ###
def split_data_labels(csv_path):
    """ TODO """
    filenames = []
    labels = []
    with open(csv_path, 'r') as f:
        for line in f:
            new_line = line.strip().split(',')
            filenames.append(new_line[0])
            labels.append(new_line[1])
    return filenames,labels

def preprocess_img(filename, label):
    """ TODO """
    image_string = tf.read_file(filename)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [64, 64])
    return resized_image, label

def img_augmentation(image, label):
    """ TODO """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def build_dataset(data, labels):
    """ TODO """
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(len(data))
    dataset = dataset.map(preprocess_img, num_parallel_calls=4)
    dataset = dataset.map(img_augmentation, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

def main():
    #### ---- ConfigParse Utility ---- ####
    config = ConfigParser()
    config.read('../../config/data_path.ini')

    try:
        data_path = config.get('data', 'data_path')
        train_paths = config.get('sample', 'train_sample')
        valid_paths = config.get('sample', 'valid_sample')
        # train_paths = config.get('csv', 'csv_path') + 'train.csv'
        # valid_paths = config.get('csv', 'csv_path') + 'valid.csv'
        print("Your data path is set to: '{}'".format(data_path))
    except:
        print('could not read data_path.ini file. Make sure you paths are correct.')
        sys.exit(1)

    # Generate seperate lists of img paths and labels to feed into tf.data
    train_imgs, train_labels = split_data_labels(train_paths)
    valid_imgs, valid_labels = split_data_labels(valid_paths)

    sys.exit()




if __name__ == '__main__':
    main()


#### examples
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.shuffle(len(filenames))
dataset = dataset.map(parse_function, num_parallel_calls=4)
dataset = dataset.map(train_preprocess, num_parallel_calls=4)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next() #graph node to contain next element of iterator
init_op = iterator.initializer #huh??

with tf.Session() as sess:
    # Initialize the iterator
    sess.run(init_op)
    print(sess.run(next_element))
    print(sess.run(next_element))
    # Move the iterator back to the beginning
    sess.run(init_op)
    print(sess.run(next_element))

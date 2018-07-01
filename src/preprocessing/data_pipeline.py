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




###### ------------- testing -------------- ######
# dataset = tf.data.TextLineDataset(train_paths) # node in the tf graph
# dataset = dataset.shuffle(buffer_size=len(dataset))
# dataset = dataset.batch(3)
# # iterator = dataset.make_one_shot_iterator() #iterator to iter over dataset once.
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next() #graph node to contain next element of iterator
# init_op = iterator.initializer #huh??
#
# with tf.Session() as sess:
#     # Initialize the iterator
#     sess.run(init_op)
#     print(sess.run(next_element))
#     print(sess.run(next_element))
#     # Move the iterator back to the beginning
#     sess.run(init_op)
#     print(sess.run(next_element))
#
# sys.exit()
###### ------------- testing -------------- ######

def split_data_labels(csv_path):
    filenames = []
    labels = []
    with open(csv_path, 'r') as f:
        for line in f:
            new_line = line.strip().split(',')
            filenames.append(new_line[0])
            labels.append(new_line[1])
    return filenames,labels

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









train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(1)             # skip the first header row
train_dataset = train_dataset.map(parse_csv)      # parse each row
train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
train_dataset = train_dataset.batch(32)

# View a single example entry from a batch
features, label = iter(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])
# example features: tf.Tensor([6.  2.7 5.1 1.6], shape=(4,), dtype=float32)
# example label: tf.Tensor(1, shape=(), dtype=int32)



# need a function to go grab those csv files and pull them in
def injest_csv():
    pass

def create_data():
    pass





# Toy data
train_imgs = tf.constant(['train/img431.png', 'train/img2.png',
                          'train/img3.png', 'train/img4.png',
                          'train/img5.png', 'train/img6.png'])
train_labels = tf.constant([0, 0, 0, 1, 1, 1])

val_imgs = tf.constant(['val/img1.png', 'val/img2.png',
                        'val/img3.png', 'val/img43.png'])
val_labels = tf.constant([0, 0, 1, 1])

# create TensorFlow Dataset objects
# tr_data = Dataset.from_tensor_slices((train_imgs, train_labels))
# val_data = Dataset.from_tensor_slices((val_imgs, val_labels))
tr_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
val_data = tf.data.Dataset.from_tensor_slices((val_imgs, val_labels))

# create TensorFlow Iterator object
iterator = Iterator.from_structure(tr_data.output_types,
                                   tr_data.output_shapes)
next_element = iterator.get_next()

# create two initialization ops to switch between the datasets
training_init_op = iterator.make_initializer(tr_data)
validation_init_op = iterator.make_initializer(val_data)

with tf.Session() as sess:

    # initialize the iterator on the training data
    sess.run(training_init_op)

    # get each element of the training dataset until the end is reached
    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break

    # initialize the iterator on the validation data
    sess.run(validation_init_op)

    # get each element of the validation dataset until the end is reached
    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break

if __name__ == '__main__':
    main()

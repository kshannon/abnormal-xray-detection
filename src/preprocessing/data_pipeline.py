#!/usr/bin/env python

# Script that creates tensorflow data.Dataset object and pickles them
# Run this once to generate the pickle files. Make sure your paths are
# set correctly in the data_path.ini file. The paths should lead to your
# MURA .csv files

# !!! When saving a model's weights, tf.keras defaults to the checkpoint format. Pass save_format='h5' to use HDF5.


import sys
import os
import argparse
from configparser import ConfigParser
import pathlib
from glob import glob
import tensorflow as tf
from tensorflow import keras
# from tensorflow.contrib.data import Data, Iterator
tf.enable_eager_execution()
tfe = tf.contrib.eager


### # TODO: This script should become modular, will rip out the config parser
# and add that to train model, which will import this script, feed this script's
# functions the file paths etc etc TODO


### --- Globals --- ###
BATCH_SIZE = 3
PREFETCH_SIZE = 1
sample = True
img_resize_x = 64
img_resize_y = 64


### --- Helper Function --- ###
def split_data_labels(csv_path, data_path):
    """ TODO """
    filenames = []
    labels = []
    with open(csv_path, 'r') as f:
        for line in f:
            new_line = line.strip().split(',')
            filenames.append(data_path + new_line[0])
            labels.append(float(new_line[1]))
    return filenames,labels

def save_img(img, label):
    """ TODO """
    # TODO take an img or imgs from a batch and save them with the class num in the name
    pass

def preprocess_img(filename, label):
    """ TODO """
    image_string = tf.read_file(filename)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [img_resize_x, img_resize_y])
    return image, label # resized_image ??

def img_augmentation(image, label):
    """ TODO """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    print('------------------------------')
    print(type(image))
    print(image)
    print(type(label))
    print(label)
    print('------------------------------')
    # print(image)
    # sys.exit()
    return image, label

def build_dataset(data, labels):
    """ TODO """
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(len(data))
    dataset = dataset.map(preprocess_img, num_parallel_calls=4)
    dataset = dataset.map(img_augmentation, num_parallel_calls=4)
    dataset = dataset.batch(BATCH_SIZE) # (?, x, y) unknown batch size because the last batch will have fewer elements.
    # dataset = dataset.prefetch(PREFETCH_SIZE) #single training step consumes n elements
    print('------------------------------')
    print(type(dataset))
    print(dataset)
    print('------------------------------')
    dataset = dataset.repeat()
    return dataset

def main():
    #### ---- ConfigParse Utility ---- ####
    config = ConfigParser()
    config.read('../../config/data_path.ini')

    try:
        sample_data = config.get('sample', 'sample_data') #build sample data
        complete_data = config.get('data', 'data_path')
        print("Data path: '{}'".format(complete_data))
    except:
        print('Error reading data_path.ini, try checking data paths.')
        sys.exit(1)

    # Check if we are working with sample or complete data... TODO ugly...
    if sample == True:
        train_paths = sample_data + 'train.csv'
        valid_paths = sample_data + 'valid.csv'
    else:
        train_paths = complete_data + 'MURA-v1.1/train.csv'
        valid_paths = complete_data + 'MURA-v1.1/valid.csv'

    # Generate seperate lists of img paths and labels to feed into tf.data
    train_imgs, train_labels = split_data_labels(train_paths, complete_data)
    valid_imgs, valid_labels = split_data_labels(valid_paths, complete_data)

    # Build tf.data objects to interact with tf.iterator
    train_dataset = build_dataset(train_imgs, train_labels) #training data
    valid_dataset = build_dataset(valid_imgs, valid_labels) #validation data
    print(train_dataset)

    # train a simple sample model
    model = keras.Sequential([
    keras.layers.Conv2D(64, kernel_size=(3,3), input_shape=(64, 64, 3), data_format="channels_last"),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])



    model.fit(train_dataset, epochs=10, steps_per_epoch=30,
        validation_data=valid_dataset,
        validation_steps=30)
    sys.exit()





    # Using Eager Execution and no Session
    # for img, label in train_dataset:
    #     print(img)
    #     print(label)

    # # train a simple sample model
    # model = keras.Sequential([
    # keras.layers.Dense((img_resize_x * img_resize_y), activation=tf.nn.relu, input_shape = (64, 64, 3)),
    # keras.layers.Dense(2, activation=tf.nn.softmax)
    # ])
    # model.add(Flatten())
    #
    # model.compile(optimizer=tf.train.AdamOptimizer(),
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy'])
    #
    # model.fit(train_dataset, epochs=1, steps_per_epoch=2)
    #
    # valid_loss, valid_acc = model.evaluate(test_images, test_labels)
    #
    # print('Test accuracy:', test_acc)

    # iterator = train_dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()
    # print(type(iterator)) #<class 'tensorflow.python.data.ops.iterator_ops.Iterator'>
    # print(next_element) #(<tf.Tensor 'IteratorGetNext:0' shape=(?, 64, 64, 3) dtype=float32>, <tf.Tensor 'IteratorGetNext:1' shape=(?,) dtype=string>)


    # with tf.Session() as sess:
    #     count = 0
    #     while True:
    #         try:
    #             elem = sess.run(next_element)
    #             print(elem[0].shape)
    #             print(elem[1])
    #             count += 1
    #         except tf.errors.OutOfRangeError:
    #             print(count)
    #             print("End of training dataset.")
    #             break





    # # DEBUG print statements
    # print(valid_dataset) #<PrefetchDataset shapes: ((?, 64, 64, 3), (?,)), types: (tf.float32, tf.string)>
    # print(type(valid_dataset)) #<class 'tensorflow.python.data.ops.dataset_ops.PrefetchDataset'>
    # print(valid_dataset.output_classes) #(<class 'tensorflow.python.framework.ops.Tensor'>, <class 'tensorflow.python.framework.ops.Tensor'>)
    # print(valid_dataset.output_shapes) #(TensorShape([Dimension(None), Dimension(64), Dimension(64), Dimension(3)]), TensorShape([Dimension(None)]))
    # print(valid_dataset.output_types) #(tf.float32, tf.string)

    sys.exit()



if __name__ == '__main__':
    main()

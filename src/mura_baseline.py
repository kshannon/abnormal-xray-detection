# baseline DenseNet model from Andrew Ng's MURA paper:

# Rajpurkar, P., Irvin, J., Bagul, A., Ding, D., Duan, T.,
# Mehta, H., ... & Langlotz, C. (2017). Mura dataset: Towards
# radiologist-level abnormality detection in musculoskeletal
# radiographs. arXiv preprint arXiv:1712.06957.

# This code attempts to recreate the results from the MURA paper.

#### NOTES.........
#[1]
'''The model takes as input one or more views for a study. On each view, our 169-layer
convolutional neural network predicts the probability of abnormality; the per-view probabilities are
then averaged to output the probability of abnormality for the study.'''
#[2]
'''final fully connected layer has a single output, after which we applied a sigmoid nonlinearity.'''
#[3]
'''Before feeding images into the network, we normalized each image to have the same mean and
standard deviation of images in the ImageNet training set. We then scaled the variable-sized images
to 320 × 320. We augmented the data during training by applying random lateral inversions and
rotations of up to 30 degrees.'''
#[4]
'''The weights of the network were initialized with weights from a model pretrained on ImageNet
(Deng et al., 2009). The network was trained end-to-end using Adam with default parameters β1 =
0.9 and β2 = 0.999 (Kingma & Ba, 2014).'''
#[5]
'''We trained the model using minibatches of size 8. We
used an initial learning rate of 0.0001 that is decayed by a factor of 10 each time the validation loss
plateaus after an epoch. We ensembled the 5 models with the lowest validation losses.'''


#### ========= Import Statements ========= ####
import sys
import os
import argparse
from configparser import ConfigParser
import pathlib
from glob import glob
import tensorflow as tf
from tensorflow import keras
tf.enable_eager_execution()
tfe = tf.contrib.eager




#### ========= Argparse Utility ========= ####
parser = argparse.ArgumentParser(description='Modify the MURA Baseline Script',add_help=True)
parser.add_argument('-max_data',
                    action="store_true",
                    dest="max_data",
                    default=False,
                    help='Set True to train on full dataset')
args = parser.parse_args()


#### ========= ConfigParse Utility ========= ####
config = ConfigParser()
config.read('../config/data_path.ini')


#### ========= Global Vars ========= ####
PREFETCH_SIZE = 1
MAX_DATA = args.max_data
img_resize_x = 320
img_resize_y = 320
MINIBATCH_SIZE = 8
LEARNING_RATE = 0.0001
DECAY_FACTOR = 10
ADAM_B1 = 0.9
ADAM_B2 = 0.999


#### ========= Ingest Data ========= ####
try:
    sample_data = config.get('sample', 'sample_data')
    max_data = config.get('data', 'data_path')
except:
    print('Error reading data_path.ini, try checking data paths.')
    sys.exit(1)

# Check if we are working with sample or complete data... TODO ugly...
if MAX_DATA == False:
    train_paths = sample_data + 'train.csv'
    valid_paths = sample_data + 'valid.csv'
    print("Using SAMPLE dataset. Data path: '{}'".format(sample_data))
else:
    train_paths = max_data + 'MURA-v1.1/train.csv'
    valid_paths = max_data + 'MURA-v1.1/valid.csv'
    print("Using Full dataset. Data path: '{}'".format(max_data))


#### ========= Helper Functions ========= ####
def normalize_data():
    pass





def main():
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


if __name__ == '__main__':
    main()

# # Generate seperate lists of img paths and labels to feed into tf.data
# train_imgs, train_labels = split_data_labels(train_paths, complete_data)
# valid_imgs, valid_labels = split_data_labels(valid_paths, complete_data)
#
# # Build tf.data objects to interact with tf.iterator
# train_dataset = build_dataset(train_imgs, train_labels) #training data
# valid_dataset = build_dataset(valid_imgs, valid_labels) #validation data
# print(train_dataset)

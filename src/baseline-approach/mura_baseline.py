# baseline DenseNet approach from Andrew Ng's MURA paper:

# Rajpurkar, P., Irvin, J., Bagul, A., Ding, D., Duan, T.,
# Mehta, H., ... & Langlotz, C. (2017). Mura dataset: Towards
# radiologist-level abnormality detection in musculoskeletal
# radiographs. arXiv preprint arXiv:1712.06957.

# This code attempts to recreate the results from the MURA paper.



#### ========= Import Statements ========= ####
import sys
import os
import time
from random import randint
import argparse
from configparser import ConfigParser
import h5py
import pathlib
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
# tf.enable_eager_execution()
# tfe = tf.contrib.eager

#### ========= Argparse Utility ========= ####
parser = argparse.ArgumentParser(description='Modify the MURA Baseline Script',add_help=True)
parser.add_argument('-max_data',
                    action="store_true",
                    dest="max_data",
                    default=False,
                    help='Pass "max_data" to train on full MURA dataset')
parser.add_argument('-model_summary',
                    action="store_true",
                    dest="model_summary",
                    default=False,
                    help='Pass "model_summary" to print details about DenseNet169 layers')
args = parser.parse_args()


#### ========= ConfigParse Utility ========= ####
config = ConfigParser()
config.read('../../config/data_path.ini')
try:
    sample_data = config.get('sample', 'sample_data')
    max_data = config.get('data', 'data_path')
except:
    print('Error reading data_path.ini, try checking data paths in the .ini')
    sys.exit(1)

#### ========= Global Vars and Constants ========= ####
MAX_DATA = args.max_data
MODEL_SUMMARY = args.model_summary
EPOCHS = 30
IMG_RESIZE_X = 320
IMG_RESIZE_Y = 320
CHANNELS = 3
BATCH_SIZE = 8
PREFETCH_SIZE = 1
LEARNING_RATE = 0.0001
# DECAY_FACTOR = 10 #learnng rate decayed when valid. loss plateaus after an epoch
ADAM_B1 = 0.9 #adam optimizer default beta_1 value (Kingma & Ba, 2014)
ADAM_B2 = 0.999 #adam optimizer default beta_2 value (Kingma & Ba, 2014)
MAX_ROTAT_DEGREES = 30 #up to 30 degrees img rotation.
MIN_ROTAT_DEGREES = 0
TB_LOG_DIR = "./TensorBoard_logs"
CHECKPOINT_FILENAME = "./DenseNet169_baseline_{}.hdf5".format(time.strftime("%Y%m%d_%H%M%S"))# Save Keras model to this file
MODEL_FILENAME = "./DenseNet169_baseline_model"

if MAX_DATA == False:
    BATCH_SIZE = 4
    VALIDATION_STEPS = 2


#### ========= Ingest Data ========= ####
if MAX_DATA == False:
    train_paths = sample_data + 'train.csv'
    valid_paths = sample_data + 'valid.csv'
    data_path = max_data
    print("Using SAMPLE dataset. Data path: '{}'".format(sample_data))
else:
    train_paths = max_data + 'MURA-v1.1/train.csv'
    valid_paths = max_data + 'MURA-v1.1/valid.csv'
    data_path = max_data
    print("Using Full dataset. Data path: '{}'".format(max_data))

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

train_imgs, train_labels = split_data_labels(train_paths, data_path)
valid_imgs, valid_labels = split_data_labels(valid_paths, data_path)


#### ========= Helper Functions ========= ####
def preprocess_img(filename, label):
    """
    Read filepaths and decode into numerical tensors
    Ensure img is the required dim and has been normalized to ImageNet mean/std
    """
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=CHANNELS) # Don't use tf.image.decode_image
    image = tf.image.convert_image_dtype(image, tf.float32) #convert to float values in [0, 1]
    image = tf.image.resize_images(image, [IMG_RESIZE_X, IMG_RESIZE_Y])
    return image, label


def img_augmentation(image, label):
    """ Call this on minibatch at time of training """
    image = tf.image.random_flip_left_right(image) #lateral inversion with P(0.5)
    image = tf.image.rot90(image, k=randint(0, 4)) #not 0-30 degrees, but 90 degree increments... so sue me!
    #TODO # rotatse up to 30 https://www.tensorflow.org/api_docs/python/tf/contrib/image/rotate
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/random_rotation
    image = tf.clip_by_value(image, 0.0, 1.0) #ensure [0.0,1.0] img constraint
    return image, label

def build_dataset(data, labels):
    """todo"""
    labels = tf.one_hot(tf.cast(labels, tf.uint8), 1) #cast labels to dim 2 tf obj
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(len(data)) #mioght not need this.... DEBUG
    dataset = dataset.repeat()
    dataset = dataset.map(preprocess_img, num_parallel_calls=2) 
    dataset = dataset.map(img_augmentation, num_parallel_calls=2)
    dataset = dataset.batch(BATCH_SIZE) # (?, x, y) unknown batch size because the last batch will have fewer elements.
    dataset = dataset.prefetch(PREFETCH_SIZE) #single training step consumes n elements
    return dataset


#### ========= Training & Fitting ========= ####
def main():
    with tf.device('/cpu:0'):
        print("Building Train/Validation Dataset Objects")
        train_dataset = build_dataset(train_imgs, train_labels)
        valid_dataset = build_dataset(valid_imgs, valid_labels)

    print("Downloading DenseNet PreTrained Weights... Might take ~0:30 seconds")
    DenseNet169 = tf.keras.applications.densenet.DenseNet169(include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(IMG_RESIZE_X, IMG_RESIZE_Y, CHANNELS),
            pooling='max',
            classes=2)
    last_layer = DenseNet169.output
    # print(last_layer)
    preds = tf.keras.layers.Dense(1, activation='sigmoid')(last_layer)
    model = tf.keras.Model(DenseNet169.input, preds)

    # https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,
            beta1=ADAM_B1,
            beta2=ADAM_B2)
    
    optimizer_keras = tf.keras.optimizers.Adam(lr=LEARNING_RATE,
            beta_1=ADAM_B1,
            beta_2=ADAM_B2,
            decay=0.10)

    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_FILENAME,
            monitor="val_loss",
            verbose=1,
            save_best_only=True)

    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TB_LOG_DIR,
            # histogram_freq=1, #this screwed us over... caused tensorboard callback to fail.. why??? DEBUG !!!!!!
            # batch_size=BATCH_SIZE, # and take this out... and boom.. histogam frequency works. sob
            write_graph=True,
            write_grads=False,
            write_images=True)

    print("Compiling Model!")
    model.compile(optimizer=optimizer_keras,
            loss='binary_crossentropy',
            metrics=['accuracy'])

    if MODEL_SUMMARY == True:
        print("Printing Details about DenseNet169")
        print(model.summary()) # details about model's layers

    print("Beginning to Train Model")
    model.fit(train_dataset,
            epochs=EPOCHS,
            steps_per_epoch=(len(train_labels)//BATCH_SIZE), #36808 train number
            verbose=1,
            validation_data=valid_dataset,
            validation_steps= (len(valid_labels)//BATCH_SIZE),  #3197 validation number
            callbacks=[checkpointer,tensorboard])  #https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
            
        
    # Save entire model to a HDF5 file
    model.save(MODEL_FILENAME + '.h5')
    # # Recreate the exact same model, including weights and optimizer.
    # model = keras.models.load_model('my_model.h5')
    sys.exit()


if __name__ == '__main__':
    main()

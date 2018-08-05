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

#### TODOs.......
#TODO [1]
# baseline model makes predictions based on arithematic mean of a view...
# might need to turn split_data_labels() into a dict to capture the view info?
# or only for inference step.... think about this.
#TODO [2]
# normalize to ImageNet mean/std ?? found this online?? [ 103.939, 116.779, 123.68 ]
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
#TODO [3]
# add model checkpointing ability, arg parse and fileself.
#TODO [4]
# add print to logging: https://docs.python.org/3.7/library/logging.html

# ==================================  CODE  ==================================

#### ========= Import Statements ========= ####
import sys
import os
import time
import argparse
from configparser import ConfigParser
import h5py
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
                    help='Pass "max_data" to train on full MURA dataset')
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
IMG_RESIZE_X = 320
IMG_RESIZE_Y = 320
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
DECAY_FACTOR = 10 #learnng rate decayed when valid. loss plateaus after an epoch
ADAM_B1 = 0.9 #adam optimizer default beta_1 value (Kingma & Ba, 2014)
ADAM_B2 = 0.999 #adam optimizer default beta_2 value (Kingma & Ba, 2014)
MAX_ROTAT_DEGREES = 30 #up to 30 degrees img rotation.
MIN_ROTAT_DEGREES = 0
CHECKPOINT_FILENAME = "./DenseNet169_baseline{}".format(HOLDOUT_SUBSET) + \
                        time.strftime("_%Y%m%d_%H%M%S") + \
                        ".hdf5" # Save Keras model to this file


#### ========= Ingest Data ========= ####
if MAX_DATA == False:
    train_paths = sample_data + 'train.csv'
    valid_paths = sample_data + 'valid.csv'
    data_path = sample_data
    print("Using SAMPLE dataset. Data path: '{}'".format(sample_data))
else:
    train_paths = max_data + 'MURA-v1.1/train.csv'
    valid_paths = max_data + 'MURA-v1.1/valid.csv'
    data_path = max_data
    print("Using Full dataset. Data path: '{}'".format(max_data))

def split_data_labels(csv_path, data_path):
    """ take CSVs with filepaths/labels and extracts them into lists"""
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
    image = tf.image.decode_jpeg(image_string, channels=0) # Don't use tf.image.decode_image
    #   image = tf.image.per_image_standardization(image) #norm over entire dataset instead...
    # This will convert to float values in [0, 1]
    #   image = tf.image.convert_image_dtype(image, tf.float32)
    # or do we write our own...?
    #   image = normalize_img(image)
    image = tf.image.resize_images(image, [IMG_RESIZE_X, IMG_RESIZE_Y])
    return image, label

def normalize_data():
    # using ImageNet mean and standard deviation?
    # we would just do some simple arithmetic and return the tensor
    pass

def img_augmentation(image, label):
    """ Call this on minibatch at time of training """
    # image = tf.image.random_flip_left_right(image) # %, make sure this is inversions
    # rotatse up to 30 https://www.tensorflow.org/api_docs/python/tf/contrib/image/rotate
    #TODO

    # Make sure the image is still in [0, 1] ????? Do i really need this....
    # image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def build_dataset(data, labels):
    """ TODO """
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(len(data))
    dataset = dataset.map(preprocess_img, num_parallel_calls=4) #TODO num_parallel_calls?
    # dataset = dataset.map(img_augmentation, num_parallel_calls=4)
    dataset = dataset.batch(BATCH_SIZE) # (?, x, y) unknown batch size because the last batch will have fewer elements.
    # dataset = dataset.prefetch(PREFETCH_SIZE) #single training step consumes n elements
    dataset = dataset.repeat()
    return dataset




#rotation.. https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/random_rotation


def main():
    train_dataset = build_dataset(train_imgs, train_labels)
    valid_dataset = build_dataset(valid_imgs, valid_labels)
    #DEBUG start
    # print(type(train_dataset))
    # iterator = train_dataset.make_one_shot_iterator()
    # for x, y in iterator:
    #     print(x, y)
    #     print(type(x))
    #     print(x.shape)
    #     z = x.numpy()
    #     print(z.max())
    #     print(z.min())
    #     break
    #
    #
    # sys.exit()
    # print("Downloading DenseNet PreTrained Weights...")
    # https://keras.io/applications/#densenet
    # keras.applications.densenet.DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

    model = keras.Sequential([
    keras.layers.Conv2D(64, kernel_size=(3,3), input_shape=(64, 64, 3), data_format="channels_last"),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    print("Printing Details ")
    print(model.summary()) # details about model's layers




    model.fit(train_dataset, epochs=10, steps_per_epoch=30,
        validation_data=valid_dataset,
        validation_steps=30)
    sys.exit()


if __name__ == '__main__':
    main()

# # Generate seperate lists of img paths and labels to feed into tf.data

#
# # Build tf.data objects to interact with tf.iterator
# train_dataset = build_dataset(train_imgs, train_labels) #training data
# valid_dataset = build_dataset(valid_imgs, valid_labels) #validation data
# print(train_dataset)




# checkpoint:
# import time
# # Save Keras model to this file
# CHECKPOINT_FILENAME = "./cnn_3d_64_64_3_HOLDOUT{}".format(HOLDOUT_SUBSET) + time.strftime("_%Y%m%d_%H%M%S") + ".hdf5"
#
# print(CHECKPOINT_FILENAME)
#
#     tb_log = keras.callbacks.TensorBoard(log_dir=TB_LOG_DIR,
#                                 histogram_freq=0,
#                                 batch_size=batch_size,
#                                 write_graph=True,
#                                 write_grads=True,
#                                 write_images=True,
#                                 embeddings_freq=0,
#                                 embeddings_layer_names=None,
#                                 embeddings_metadata=None)
#
#
#     checkpointer = keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_FILENAME,
#                                                    monitor="val_loss",
#                                                    verbose=1,
#                                                    save_best_only=True)

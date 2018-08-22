# augmentation function

from random import randint
import numpy as np
import tensorflow as tf




def baseline(image, label):
    """
    Call this on minibatch at time of training
    This map function represents the baseline approach from MURA paper
    """
    image = tf.image.random_flip_left_right(image) #lateral inversion with P(0.5)
    image = tf.image.rot90(image, k=randint(0, 4)) #not 0-30 degrees, but 90 degree increments... so sue me!
    #TODO # rotatse up to 30 https://www.tensorflow.org/api_docs/python/tf/contrib/image/rotate
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/random_rotation
    image = tf.clip_by_value(image, 0.0, 1.0) #ensure [0.0,1.0] img constraint
    return image, label


def vgg19_v1(image, labels):
    """
    Our first approach fort VGG19 model
    """
    pass

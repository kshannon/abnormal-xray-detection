#!/usr/bin/python

# Inference Script for MURA Stanford challenge
# cl run valid_image_paths.csv:valid_image_paths.csv MURA-v1.1:valid src:src "python src/inference.py valid_image_paths.csv predictions.csv" -n run-predictions


import sys
import os
import csv
import re
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models


#### ========= Global Vars and Constants ========= ####
model = models.load_model('src/DenseNet169_baseline_model.h5') # Load model, weights and meta data
IMG_RESIZE_X = 320
IMG_RESIZE_Y = 320
CHANNELS = 3
path_to_prediction_program = str(sys.argv[0])
input_data_csv_filename = str(sys.argv[1])
output_prediction_csv_path = str(sys.argv[2])


#### ========= Helper Functions ========= ####
def id_generator(csv_line):
    """
    Takes one line from input CSV and extracts the proper study path and
    creates a unqiue patient/study ID that we can use as a key in a dictionary
    Basically this function sets up our ability to organize the data by study
    """
    csv_line = csv_line.rstrip('\n') #chomp chomp
    split_line = csv_line.split('/') #tokenize line
    patient = split_line[3][7:] #get the patient number
    study = re.search(r'([0-9]+)',split_line[4]).group(0) #get the study number
    record = patient + '/' + study #create unique patient study record
    return csv_line, record

def strip_filename(path):
    dirname, filename = os.path.split(path)
    return dirname + '/'

def prepare_img(filename):
    """Prepare an image with the same preprocessing steps used during training (no augmentation)"""
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=CHANNELS) # Don't use tf.image.decode_image
    image = tf.image.convert_image_dtype(image, tf.float32) #convert to float values in [0, 1]
    image = tf.image.resize_images(image, [IMG_RESIZE_X, IMG_RESIZE_Y])
    image = image[np.newaxis,...] #add on that tricky batch axis
    return image

def inference(img_path, model, data_path=None):
    """Send an img and model, preprocess the img to training standards, then return a pred"""
    img = prepare_img(img_path)
    pred_prob = model.predict(img, batch_size=None, steps=1, verbose=0)
    return pred_prob[0][0]

def avg_probabilities(prob_vector):
    """Takes in a vector of probabilites for a study, avg them and return a single [0,1] prob"""
    vec = np.array(prob_vector)
    avg_prob = vec.sum()/len(prob_vector)
    return int(np.where(avg_prob > 0.5, 1, 0))


def main():
    # STEP 1: organize the data into patient study groups
    patient_dict = {} #our new data study based structure key = patient_num/study_num e.g. 11185/1, 11185/2
    with open(input_data_csv_filename,'r') as in_file:
        buffer = []
        previous_id = None
        for line in in_file:
            data, unique_id = id_generator(line) #sanitize data
            if previous_id == None: #special case for first loop
                previous_id = unique_id
            if previous_id != unique_id: #write the buffers to the dict if a new patient and or study appear
                patient_dict[previous_id] = buffer
                buffer = [] #flush buffers
                previous_id = unique_id
            buffer.append(data)

    # STEP 2: for each study group, predict on each image in the group and then determine the avg prob
    predictions = []
    for patient_study_id, img_path_list in patient_dict.items():
        prob_vector = []
        dir_path = strip_filename(img_path_list[0])
        for img_path in img_path_list:
            pred = inference(img_path, model) #i'm sure we can do this as a batch, memory contraints???
            prob_vector.append(pred)
        classification = avg_probabilities(prob_vector)
        predictions.append((dir_path, classification))

    # STEP 3: write out the avg prediction per study to a csv
    with open(output_prediction_csv_path,'w', newline='') as out_file:
        writer = csv.writer(out_file)
        for result in predictions:
            writer.writerow([result[0],result[1]])


if __name__ == '__main__':
    main()

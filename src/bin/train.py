#!/usr/bin/python

# pass as positional args a config .yml file and provide an experiment text name e.g.
# name@host: $ python train.py test.yml test.txt

# script to kick-off model training, preprocessing, and monitoring
# takes cmd line args to load in config files

import sys
import os
import csv
import re
import numpy as np
import yaml
import h5py
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras import models


#### ========= Globals, Constants, Config Ingestion ========= ####

config = '../configs/' + str(sys.argv[1])
exp_file = '../experiments/' + str(sys.argv[2]) #file path/name for experiment
exp_dump = {'test':'...'} #container for data to write out to exp folder
#read config file next
print(config)



with open("../configs/test.yml", 'r') as stream:
    try:
        print(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)



def main():
    #### ========= Write Out Experiment Information ========= ####
    with open(exp_file,'w') as out_file:
        for text in exp_dump:
            out_file.write(text)
            out_file.write('\n')



#### ========= Global Vars and Constants ========= ####
# require src/ as the src dir was zipped for submission purposes...
# model = models.load_model('src/DenseNet169_baseline_model.h5') # Load model, weights and meta data
# IMG_RESIZE_X = 320
# IMG_RESIZE_Y = 320
# CHANNELS = 3
# path_to_prediction_program = str(sys.argv[0])
# input_data_csv_filename = str(sys.argv[1])
# output_prediction_csv_path = str(sys.argv[2])



if __name__ == '__main__':
    main()

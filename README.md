# Stanford's MURA Bone X-Ray Deep Learning Competition
Raise: TODO

The goal of this project is to ...
Raise: TODO




The instructions below will get you a copy of the project up and running on your local machine for development and testing purposes.

# Table of Contents
1. [Data](#Data)
2. [Prerequisites](#Prerequisites)
3. [Steps](#Steps)
4. [Results](#Results)
5. [Authors](#Authors)

## Data
Data is openly availble from Stanford's ML Lab: https://stanfordmlgroup.github.io/competitions/mura/

Raise: TODO

## Prerequisites
A list of conda/pip environment dependencies can be found in the environments.yml file. To create a conda env with all of the dependencies run the create_conda_env.sh shell script. We are also using Tensorflow and Keras with GPU support.

## Steps
1. Download the MURA dataset and unzip it into a a location of your chosing.
2. Run the shell script **env_setup.sh** This will create the conda environment that we used to build the model.
3. Run the shell script **create_ini_files.sh** This will create a config.ini file where you will need to put a path to your data. for example my path is: /Users/keil/datasets/mura/
4. Run **merge_csv.py** to create the merged sample and full csv files. two csvs will be created in the sample_data/ directory and two csvs in your MURA data path location.
5. Run the **data_pipeline.py** file and congratz you are where we are! ...
6. more to come...
 
## Results
Raise: TODO


## Authors
- [Kyle Shannon](https://github.com/kshannon)
- [Chris Chen](https://github.com/utcsox)

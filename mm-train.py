import argparse
import os
import sys
from os import walk
from scipy.io import wavfile
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import logging
from sagemaker.tensorflow import TensorFlow
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from tensorflow.keras.layers import Dropout, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
from cfg import Config

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args, _ = parser.parse_known_args()

    config = Config('conv')

    # ... load from args.train and args.test, train a model, write model to args.model_dir.

    tf_estimator = TensorFlow(entry_point='mm-train.py', role='SageMakerRole',
                          train_instance_count=1, train_instance_type='ml.p2.xlarge',
                          framework_version='1.12', py_version='py3')
    tf_estimator.fit(config.bucket_train)
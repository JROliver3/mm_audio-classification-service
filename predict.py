import os
import logging
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import boto3
import traceback
from os import walk
import shutil
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from tensorflow.keras.layers import Dropout, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
from cfg import Config
from sklearn.metrics import accuracy_score


def build_predictions(audio_dir):
    y_true = []
    y_pred = []
    #output of softmax layer is (1,10) array, values totalling to 1..
    #so to find probability of ex. acoustics, find the 0th index of the (1,10) array
    fn_prob = {}
    print("Extracting Features from Audio" )
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        # label = fn2class[fn]
        # c = classes.index(label)
        y_prob = []
        config.step = int(rate/10)

        for i in range(0, wav.shape[0] - config.step, config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat, 
                     nfilt = config.nfilt, nfft = config.nfft)
            #normalize x
            x = (x - config.min)/(config.max - config.min)
            
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode == 'time':
                #expand dimensions of some array x and expanding into the 0th axis (the first parameter of the shape )
                x = np.expand_dims(x, axis=0)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
        
        fn_prob[fn]= np.mean(y_prob, axis=0).flatten()
        
    return y_pred, fn_prob

def write_single_to_csv(filename):
    with open('predict.csv', 'w') as csvfile:
        fieldnames = ['fname']
        filewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        filewriter.writeheader()
        filewriter.writerow({'fname':filename})
    print("successfully added audio to predict csv")

def download_audio_file(fn):
    bucket_name = 'mm-acs-audio-storage'
    s3_file_path= 'asc_data/audio' + '/' + fn + '.wav'
    s3_data_path = 'asc_data/data.csv'
    save_as = fn + '.wav'

    s3 = boto3.client('s3')
    s3.download_file(bucket_name, s3_file_path, save_as)
    s3.download_file(bucket_name,s3_data_path, 'data.csv')
    if not os.path.isfile(os.path.join('asc_data','audio', save_as)):
        if not os.path.exists(os.path.join('asc_data','audio')):
            os.makedirs(os.path.join('asc_data','audio'))
        shutil.move(save_as, os.path.join('asc_data','audio'))

def get_downloaded_model_data():
    df = pd.read_csv('data.csv')
    return float(df['min']), float(df['max'])


def build_single_prediction(fn, audio_dir):
    try:
        y_true = []
        y_pred = []
        fn_prob = {}
        config = Config('conv')
        download_audio_file(fn)
        write_single_to_csv(fn)
        _min, _max = get_downloaded_model_data()
        model = load_model(config.model_path)
        print("Extracting Features from Audio with model data min: " )

        rate, wav = wavfile.read(os.path.join(audio_dir, fn+'.wav'))
        y_prob = []
        config.step = int(rate/10)

        for i in range(0, wav.shape[0] - config.step, config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat, 
                        nfilt = config.nfilt, nfft = config.nfft)
            #normalize x
            x = (x - _min)/(_max - _min)
            
            x = x.reshape(1, x.shape[0], x.shape[1], 1)
        
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))

        fn_prob[fn]= np.mean(y_prob, axis=0).flatten()
        
        y_probs, classes = get_probabilities(fn_prob)

        y_pred = [classes[np.argmax(y)] for y in y_probs]

        remove_acs_data_dir()

        return y_pred[0]
    except:
        return traceback.format_exc()

def remove_acs_data_dir():
    shutil.rmtree('asc_data')

def get_dataframes():
    df = pd.read_csv('predict.csv')
    dfcats = pd.read_csv('cats.csv')
    return df, dfcats

def get_probabilities(fn_prob):
    df, dfcats = get_dataframes()
    classes = list(np.unique(dfcats.label)) 

    y_probs = []
    for i, row in df.iterrows():
        y_prob = fn_prob[row.fname]
        y_probs.append(y_prob)
        for c,p in zip(classes, y_prob):
            df.at[i, c] = p
    return y_probs, classes

if __name__ == "__main__":
    #the info given in instruments.csv will have to be derived from directory labels with new data
    p_path = os.path.join('pickles', 'conv.p')

    with open(p_path, 'rb') as handle:
        config = pickle.load(handle)

    df, dfcats = get_dataframes()
    classes = list(np.unique(dfcats.label))    

    model = load_model(config.model_path)

    y_pred, fn_prob = build_predictions('predict')

    y_probs = get_probabilities(fn_prob)

    y_pred = [classes[np.argmax(y)] for y in y_probs]
    df['y_pred'] = y_pred
    df.to_csv('predictions2.csv', index=False)

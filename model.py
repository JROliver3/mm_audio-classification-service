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
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from tensorflow.keras.layers import Dropout, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
from cfg import Config

def check_data():
    #check if pickle has data, isfile checks if path contains a file..
    #tmp is later checked in build_rand_feat to see if it has previously loaded data
    #this enables for smoother tweaking of models when dealing with large data stores
    if os.path.isfile(config.p_path):
        print("Loading existing data for {} model".format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    X = []
    y = []
    #for neural networks, want to normalize input between 0 and 1, therefore we need to know
    # the initial min and max to scale our values down
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        try:
            rand_class = np.random.choice(class_dist.index, p=prob_dist)
            file = np.random.choice(df[df.label == rand_class].index)
            rate, wav = wavfile.read('clean/'+file)
            label = df.at[file, 'label']
            if type(label) is not str:
                label = label[0]
            #finding a random index and stopping config.step distance away to avoid running over by the step amount
            step = int(rate/10)
            rand_index = np.random.randint(0, wav.shape[0] - step)
            sample = wav[rand_index:rand_index + step]
            X_sample = mfcc(sample, rate,
                            numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
            _min = min(np.amin(X_sample), _min)
            _max = max(np.amax(X_sample), _max)
            X.append(X_sample)
            y.append(classes.index(label))
            # testing
            config.min = _min
            config.max = _max
        except Exception as e:
            logging.exception("AudioMFCCSamplingError")
            print("wav shape: " + str(wav.shape[0]))
            print("rate: " + str(rate))
            print("step: " + str(step))
    config.min = _min
    config.max = _max
    #hot-encoding the target variables y into a matrix so Keras can handle it in the dnn
    #will need to map them back to their original indices later 
    X, y = np.array(X, np.float64), np.array(y, np.float64)
    X = (X - _min)/(_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        #the shape has three dimensions, n_samples, time dimension, and our mscc features
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=n_classes)
    config.data = (X, y)
    
    #open a handle in the pickle path (stored as p_path in the config)
    #dump the entire current config into the pickle
    #later you will attempt to access this pickle using tmp in the check_data method
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
        
    return X,y

def get_conv_model():
    #if input space is really high, having a lot of pooling makes sense. (?)
    #because we're only using 1/10 of a second and the matrix is only (13, 9, 1)
    model = Sequential()
    #if you have huge input space, consider making stride (2,2) and the first kernel convolution (5,5) instead of (3,3)
    model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1), 
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1),
                     padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1),
                     padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1),
                     padding='same'))
    #the more convolutional layers, the more features you can find about the data, and the number of filters
    #increases to become more granular as it becomes convolved down between each layer
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    #the activation function is softmax because of categorical cross entropy
    #we pulled down the dense layers to our 10 class activation
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    #because we're doing classification, our loss will be 
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['acc'])
    return model

def get_recurrent_model():
    #shape of data for rnn is (n, time, feat).. this was the reason for the transpose, to shape the data correctly
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    #can't use model.add(Dense) due to using sequences here
    #instead we can use a time distributed layer
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['acc'])
    return model

#need to get instrument labels here and build the csv
config = Config(mode='conv')

df = pd.read_csv('cats.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    try:
        rate, signal = wavfile.read('clean/'+f)
        df.at[f, 'length'] = signal.shape[0]/rate
    except:
        print(sys.exc_info()[0])

classes = list(np.unique(df.label))
n_classes = len(classes)
class_dist = df.groupby(['label'])['length'].mean()

n_samples = 2 * int(df['length'].sum()/0.1)
prob_dist = class_dist / class_dist.sum()
random_choice = np.random.choice(class_dist.index, p=prob_dist)

# fig, ax = plt.subplots()
# ax.set_title('Class Distribution', y=1.08)
# ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
#        shadow=False, startangle=90)
# ax.axis('equal')
# plt.show()

if config.mode == 'conv':
    X, y = build_rand_feat()
    #mapping the "hot-encodedy matrix and the values back to their original class encoding (the indices that go with the labels)
    #maps back to the original column when identifying a "1" vs a "0".
    y_flat = np.argmax(y, axis=1)
    #whenever giving an input shape to the first layer of a neural network we don't count the n_samples:
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()
    
elif config.mode == 'time':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()

#this is to account for the low probability distribution of some attributes and define a weight
#that compensates in order to learn learn probability features just as well.
#this method adds just a bit more accuracy
class_weight = compute_class_weight('balanced',
                                    np.unique(y_flat),
                                    y_flat)

#for monitor we have acc, validation acc, loss, val_loss
#if loss, then you would set the mode to min
checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)
#randomly create batches of our data
model.fit(X, y, epochs=10, batch_size=32,
          shuffle=True, validation_split=0.1,
          class_weight=class_weight, callbacks =[checkpoint])

model.save(config.model_path)
    







import os
import sys
from os import walk
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wavfile
import logging
from python_speech_features import mfcc, logfbank
from pydub import AudioSegment
import librosa as librosa
from cfg import Config

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=config.plot_rows, ncols=config.plot_cols, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(config.plot_rows):
        for y in range(config.plot_cols):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=config.plot_rows, ncols=config.plot_cols, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(config.plot_rows):
        for y in range(config.plot_cols):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=config.plot_rows, ncols=config.plot_cols, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(config.plot_rows):
        for y in range(config.plot_cols):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=config.plot_rows, ncols=config.plot_cols, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(config.plot_rows):
        for y in range(config.plot_cols):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def calc_fft(y, rate):
    n = len(y)
    #npm.fft is the fourier transform, and rfft is the real frequency fft 
    freq = np.fft.rfftfreq(n, d = 1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return(Y, freq)

def envelope(y, rate, threshold):
    mask = []
    # convert from numpy array to series, then create absolute value of it
    y = pd.Series(y).apply(np.abs)
    #panda series can do rolling window over your data, another benefit to 
    #using panda series
    #this will look at a window of values as opposed to single values
    #that would result in a loss of a lot of data if we removed them
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if(mean > threshold):
            mask.append(True)
        else:
            mask.append(False)
    return mask

def find_wavdir(index):
    for (root, subdirs, filenames) in walk(config.train_dir):
        for search in subdirs:
            path = root + '/' + search
            for (_, _, filenames) in walk(path):
                for filename in filenames:
                    if index == filename:
                        wavdir = path + '/' + index
                        return wavdir

        
config = Config('conv')

df = pd.read_csv(config.train_cats +'.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    wavdir = find_wavdir(f)
    try:  
        rate, signal, _ = wavfile.read(wavdir)
        df.at[f, 'length'] = signal.shape[0]/rate
    except:
        print(sys.exc_info()[0])
    #gives the legnth of the signal in terms of seconds

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct = '%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
# plt.show()
df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
    ##wav_file = df[df.lavel == c].iloc[0,0] gets item from 0th row and 0th column of dataframe
    wav_file = df[df.label == c].iloc[0,0]
    wavdir = find_wavdir(wav_file)
    signal, rate = librosa.load(wavdir, sr=44100)
    mask = envelope(signal, rate, 0.0005)
    signal = signal[mask]
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)
    
    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    mfccs[c] = mel
    
# plot_signals(signals)
# plt.show()

# plot_fft(fft)
# plt.show()

# plot_fbank(fbank)
# plt.show()

# plot_mfccs(mfccs)
# plt.show()

if not os.path.exists('clean'):
    os.makedirs('clean')
if len(os.listdir('clean')) == 0:
    for f in tqdm(df.fname):
        try:
            wavdir = find_wavdir(f)
            # signal, rate = librosa.load(wavdir, sr=22050)
            _, signal, _ = wavfile.read(wavdir)
            rate = 22050
            # mask = envelope(signal, rate, 0.0005)
            wavfile.write('clean/'+f, rate, signal)
        except:
            logging.exception("AudioWavWritingError")
    
    
    
    
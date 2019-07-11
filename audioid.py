import boto3
import os
import csv
import sys
import shutil
import pandas as pd
from pydub import AudioSegment
from os import walk
from cfg import Config
from tqdm import tqdm

def move_audio_to_train(audio_dir, train_dir):
    df = pd.read_csv('instruments.csv')
    df.set_index('fname', inplace=True)
    for(root, _, filenames) in walk(audio_dir):
        for fname in filenames:
            label = df.at[fname, 'label']
            dest_dir = train_dir + '/' + label
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            shutil.move(root + '/' + fname, dest_dir)
            print("moving " + root + '/' + fname + " to " + dest_dir)

config = Config('conv')
move_audio_to_train('wavfiles', config.train_dir)

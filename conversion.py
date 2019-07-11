import os
from os import walk
from cfg import Config
import shutil


def remove_original_audio():
    for(root, subdirs, _) in walk(config.train_dir):
        for subdir in subdirs:
            path = root + '/' + subdir
            for(_, _, filenames) in walk(path):
                for filename in filenames:
                    if 'mmacs' not in filename:
                        os.remove(path+'/'+filename)
                        print("removing " + filename + " from " + path)


                        

config = Config('conv')
remove_original_audio()
import boto3
import os
import csv
import sys
import shutil
from pydub import AudioSegment
from os import walk
from cfg import Config
from tqdm import tqdm

# if not os.path.exists(dirName):
#     os.mkdir(dirName)
#     print("Directory " , dirName ,  " Created ")
# else:    
#     print("Directory " , dirName ,  " already exists")

config = Config('conv')

# acss3.download_file(config.bucket_n, config.ob_n, config.save_to)

acss3_client = boto3.client('s3')
acss3_resource = boto3.resource('s3')

def download_dir(prefix, sample_dir=config.save_to, bucket=config.bucket_n,
                 client=acss3_client, resource=acss3_resource):
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket':bucket,
        'Prefix':prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = acss3_client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(sample_dir, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in tqdm(keys):
        dest_pathname = os.path.join(sample_dir, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        resource.meta.client.download_file(bucket, k, dest_pathname)                        

def write_to_csv(current_dir, csv_name):
    with open(csv_name + '.csv', 'w') as csvfile:
        fieldnames = ['fname', 'label']
        filewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        filewriter.writeheader()
        for (dirpath, dirnames, _) in walk(current_dir):
            for dirname in dirnames:
                num_classes = num_classes + 1
                path = dirpath + '/' + dirname
                for (_, _, files) in walk(path):
                    for filename in files:
                        filewriter.writerow({'fname':filename, 'label':dirname})
    print("successfully created csv")

def train_data_to_wave():
    for (root, subdirs, filenames) in walk(config.train_dir):
        for search in tqdm(subdirs):
            path = root + '/' + search
            convert_path = 'convert/' + search
            for (_, _, filenames) in walk(path):
                for filename in filenames:
                    try:
                        filetype = filename.split('.')[1]
                    except:
                        print("File "+filename + " has no extension. Removing.")
                        os.remove(path+'/'+filename)
                    if filetype != 'wav':
                        # TODO: support additional filetypes
                        try:
                            filetype_dir = convert_path
                            if not os.path.exists(filetype_dir):
                                os.makedirs(filetype_dir)
                            shutil.move(path+'/'+filename, filetype_dir + '/' + filename)
                        except:
                            print("failed to send file to its folder... not sure why this happened.")
                    elif filetype != 'wav':
                        newfilename = filename.split('.')[0] + '.wav'
                        source_path = path + '/' + filename
                        source_dest = path + '/' + newfilename
                        try:
                            segment = AudioSegment.from_file(source_path, format = filetype+'f')
                            segment.export(source_dest, format = 'wav')
                        except:
                            print("File corrupt... Erasing.")
                            os.remove(source_path)

if not os.path.exists('samples'):
    download_dir('')

train_data_to_wave()

write_to_csv(config.train_dir, 'cats')

import os
import train
import conversion
from cfg import Config

def download_and_prepare_data():
    if not os.path.exists('samples'):
        train.download_dir('')
        train.train_data_to_wave()

def write_data_to_csv():
    train.write_to_csv(config.train_dir, config.train_cats)

config = Config('conv')
download_and_prepare_data()
wait = input("Convert the data using Media Encoder and press Enter to continue...")
print("Continuing...")
conversion.replace_converted_audio()
write_data_to_csv()

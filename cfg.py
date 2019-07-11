import os

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=551, rate=22050):
        self.mode = mode
        self.nfilt=nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join('opt/ml/model', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
        self.bucket_n = 'mm-acs-audio-storage'
        self.bucket_train = 's3://mm-acs-audio-storage/asc_data/train'
        self.bucket_eval = 's3://mm-acs-audio-storage/asc_data/eval'
        self.ob_n = 'asc_data/train/Flutes/afroflute_1.aif'
        self.save_to = 'samples'
        self.train_dir = 'samples/asc_data/train'
        self.convert_dir = 'convert'
        self.train_cats = 'cats'
        self.predict_dir = 'predict'
        self.predict_csv = 'predict'
        self.plot_rows = 6
        self.plot_cols = 5
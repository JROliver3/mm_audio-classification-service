In order to train the model.

1. Start by running train.py to download necessary files from the storage (if not currently downloaded. If you want to update
then delete the current folder and run again.) This will also generate a csv file.
    a. if you have new music to add, you must add it all to a folder and run audioid together with a csv that contains
    all the audio labels (currently it should be named instruments.csv).The music will likely need to be cleaned and
    encoded through Media Encoder or some other tool to wav format, mono channels. 
2. run eda.py to generate the clean folder for processing all the files, checking their encoding, and delivering them
to the clean folder for final processing.  
3. run model.py to create random features from the clean folder and shuffle and pass them through to the model for optimized training.
4. run predict.py to create a prediction csv that will be generated based on your identified instrument classifications
and give the likelihood that the audio belongs to each classification.
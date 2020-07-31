import os
from subprocess import call

# cite: https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/data/download_dataset.py
def download():
    path = os.getcwd()
    print("Downloading...")
    if not os.path.exists(path+"/datasets/data/har"):
        call(
            'wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip" -P ' + path+"/datasets/data/har",
            shell=True
        )
        print("Downloading done.\n")
    else:
        print("Dataset already downloaded. Did not download twice.\n")


    print("Extracting...")
    
    extract_directory = os.path.abspath(path+"/datasets/data/har/uci-har")
    if not os.path.exists(extract_directory):
        call(
            'unzip -nq "' +path+'/datasets/data/har/UCI HAR Dataset.zip" -d' +path+'/datasets/data/har/uci-har',
            shell=True
        )
        print("Extracting successfully done to {}.".format(extract_directory))
    else:
        print("Dataset already extracted. Did not extract twice.\n")
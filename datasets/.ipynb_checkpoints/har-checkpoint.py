"""
UCI-HAR datsets
"""
import os
import numpy as np
from datasets.download_har import download
LABELS = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
SIGNALS = ["body_acc_x_", "body_acc_y_", "body_acc_z_",
           "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
           "total_acc_x_", "total_acc_y_", "total_acc_z_"]


# taken from https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/README.md
def __load_X(X_signal_paths):
    X_signals = []

    for signal_type_path in X_signal_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_data():
    """
    Load and return the UCI-HAR dataset.

    ==============             ==============
    Training Samples total               7352
    Testing Samples total                2947 
    Number of time steps                  128
    Dimensionality                          9
    Number of targets                       6
    ==============             ==============

    # Returns
        Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test)
    """
    download()
    module_path = os.getcwd()
    train_paths = [module_path + '/datasets/data/har/uci-har/UCI HAR Dataset/train/Inertial Signals/' + signal + 'train.txt' for signal in SIGNALS]
    test_paths = [module_path + '/datasets/data/har/uci-har/UCI HAR Dataset/test/Inertial Signals/' + signal + 'test.txt' for signal in SIGNALS]

    x_train = __load_X(train_paths)
    x_test = __load_X(test_paths)
    y_train = np.loadtxt(module_path+'/datasets/data/har/uci-har/UCI HAR Dataset/train/y_train.txt',  dtype=np.int32)
    y_test = np.loadtxt(module_path+'/datasets/data/har/uci-har/UCI HAR Dataset/test/y_test.txt', dtype=np.int32)

    return (x_train, y_train), (x_test, y_test)




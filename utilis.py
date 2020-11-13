import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, Normalizer
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']


def load_UCR_data(folder='ItalyPowerDemand', xy_split=False, plot=False, expand_dims_for_conv=False):
    path_UCR = 'data/UCRArchive_2018/'
    file_train      = folder+'_TRAIN.tsv'
    file_test       = folder+'_TEST.tsv'
    full_path_train = path_UCR+folder+'/'+file_train
    full_path_test  = path_UCR+folder+'/'+file_test
    dataset_train = pd.read_csv(full_path_train, sep='\t', header=None)
    dataset_test = pd.read_csv(full_path_test, sep='\t', header=None)
    if xy_split==True:
        y_train, x_train = dataset_train.iloc[:, 0].to_numpy(), dataset_train.iloc[:, 1:].to_numpy()
        y_test,  x_test =  dataset_test.iloc[:, 0].to_numpy(), dataset_test.iloc[:, 1:].to_numpy()
        # from [1,2,1,2] to [0,1,0,1]
        le = LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)
        y_test  = le.transform(y_test)
        # normalize
        normalizer = Normalizer().fit(x_train)  # fit does nothing
        x_train = normalizer.transform(x_train)
        x_test  = normalizer.transform(x_test)
        if plot==True:
            plt.title('TRAIN data')
            for clase in np.unique(y_train):
                plt.plot(x_train[y_train==clase].transpose(), color=colors[clase], alpha=0.05)
            plt.show()
            plt.title('TEST data')
            for clase in np.unique(y_test):
                plt.plot(x_test[y_test==clase].transpose(), color=colors[clase], alpha=0.05)
            plt.show()
        if expand_dims_for_conv==True:
            x_train = np.expand_dims(x_train, axis=2)
            x_test = np.expand_dims(x_test, axis=2)
        return [x_train, y_train, x_test, y_test]
    else:
        return [dataset_train, dataset_test]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.neighbors import KernelDensity

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']



# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()





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
    
    
def read_data_blindfast(verbose=0, xy_split=False, expand_dims_for_conv=False, show_plot=True, classes=[1, 3, 4, 9], seed=42):
    """
    Parameters
    ----------
    verbose : TYPE, optional
        DESCRIPTION. The default is 0.
    xy_split : TYPE, optional
        DESCRIPTION. The default is False.
    expand_dims_for_conv : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    USA CLASES 0, 1 Y 2 SOLO CON xy_split
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    """
    
    
    filename_X = '/home/patxi/repos/GANs1D/X_all.csv'
    filename_y = '/home/patxi/repos/GANs1D/y_all.csv'
    X_raw = pd.read_csv(filename_X, header='infer', index_col=0).iloc[:, ::50].to_numpy()
    y_raw = pd.read_csv(filename_y, header='infer', index_col=0).to_numpy().squeeze()
    if verbose==1:
        for clase in np.unique(y_raw.squeeze()):
            print('clase: {} | count: {}'.format(clase, X_raw[np.where(y_raw==clase)].shape[0]))
        print('X.shape:  ', X_raw.shape)
    X = X_raw[np.where(y_raw==classes[0])]    
    y = np.array([0]*X_raw[np.where(y_raw==classes[0])].shape[0])     
    for i in range(1, len(classes)):
        X = np.vstack((
            X,
            X_raw[np.where(y_raw==classes[i])],
        ))
        y = np.concatenate((
            y,
            np.array([i]*X_raw[np.where(y_raw==classes[i])].shape[0]),
        ))
    if xy_split==True:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        if show_plot==True:
            print('TRAIN')
            for clase, cantidad in zip(np.unique(y_train, return_counts=True)[0], np.unique(y_train, return_counts=True)[1]):
                print('CLASS {}: {} samples'.format(clase, cantidad))
            print('\nTEST')
            for clase, cantidad in zip(np.unique(y_test, return_counts=True)[0], np.unique(y_test, return_counts=True)[1]):
                print('CLASS {}: {} samples'.format(clase, cantidad))
        if expand_dims_for_conv==True:
            x_train = np.expand_dims(x_train, axis=-1)
            x_test  = np.expand_dims(x_test, axis=-1)
            y_train = np.expand_dims(y_train, axis=-1)
            y_test  = np.expand_dims(y_test, axis=-1)
        return [x_train, y_train, x_test, y_test]
    else:
        return [X, y]
    
    
    
# c = 0    
# for i in range(2, 7):
# 	print('Number of elements', i)
# 	for lst in list(itertools.combinations(np.unique(y_raw), i)):
# 		print(lst)
# 		c = c+1
# print(c)



def define_clf(input_shape, n_classes=3,):
    input_data  = tf.keras.Input(shape=(input_shape))
    reshape     = tf.keras.layers.Reshape((input_shape, 1))(input_data)
    conv1D      = tf.keras.layers.Conv1D(filters=256, kernel_size=(3,), strides=2, padding='same')(reshape)
    conv1D      = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1D)
    conv1D      = tf.keras.layers.Conv1D(filters=128, kernel_size=(3,), strides=2, padding='same')(conv1D)
    conv1D      = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1D)
    conv1D      = tf.keras.layers.Conv1D(filters=128, kernel_size=(3,), strides=2, padding='same')(conv1D)
    conv1D      = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1D)
    conv1D      = tf.keras.layers.Flatten()(conv1D)                                           
    conv1D      = tf.keras.layers.Dropout(0.4)(conv1D)                                           
    out_layer   = tf.keras.layers.Dense(4, activation='sigmoid')(conv1D)
    model = tf.keras.Model(input_data, out_layer)
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def kde_estimator(X, y, random_state=None, kernel='gaussian'):
    n_classes = len(np.unique(y))
    lst_1 = [len(np.where(y==clase)[0]) for clase in range(n_classes)]
    lst_2 = [max(lst_1)-x for x in lst_1]
    X_res = np.array([]).reshape(0, X.shape[-1])
    y_res = np.array([])
    for i in range(n_classes):
        if lst_2[i]==0:
            X_res = np.concatenate([
                X_res,
                X[np.where(y==i)], 
            ])
            y_res = np.concatenate([
                y_res,
                y[np.where(y==i)],
            ])             
        else:
            print("CLASS:", i)
            kde = KernelDensity(kernel=kernel, bandwidth=0.2).fit(X[np.where(y==i)])
            X_res = np.concatenate([
                X_res,
                X[np.where(y==i)], 
                kde.sample(n_samples=lst_2[i], random_state=random_state),
            ])
            y_res = np.concatenate([
                y_res,
                y[np.where(y==i)],
                np.array([i for _ in range(lst_2[i])]),
            ])
            return X_res, y_res

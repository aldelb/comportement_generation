import numpy as np
import pickle
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import constant
import librosa
import sys

class Set(Dataset):

    def __init__(self, setType = "train"):
        # Load data
        path = constant.data_path
        datasets = constant.datasets
        X = []
        Y = []
        interval = []

        for set_name in datasets:
            current_X = []
            with open(path +'X_'+setType+'_'+set_name+'.p', 'rb') as f:
                x = pickle.load(f)
            for item in x:
                current_X.append(item[:,np.r_[constant.selected_os_index_columns]])
            current_X = np.array(current_X)

            if(len(constant.selected_os_index_columns) == 1):
                current_X = np.reshape(current_X, current_X.shape + (1,))
            if constant.derivative:
                current_X = self.addDerevative(current_X)

            with open(path +'y_'+setType+'_'+set_name+'.p', 'rb') as f:
                current_Y = pickle.load(f)

            with open(path +'intervals_test_'+set_name+'.p', 'rb') as f:
                current_interval = pickle.load(f)

            X.extend(current_X)
            Y.extend(current_Y)
            interval.extend(current_interval)
        
        X_concat = np.concatenate(X, axis=0)
        Y_concat = np.concatenate(Y, axis=0)

        x_scaler = MinMaxScaler((-1,1)).fit(X_concat)  
        y_scaler = MinMaxScaler((-1,1)).fit(Y_concat)
        X_scaled = list(map(x_scaler.transform, X))
        Y_scaled = list(map(y_scaler.transform, Y))

        self.X = X
        self.Y = Y
        self.interval = interval
        self.X_ori = X
        self.Y_ori = Y
        self.X_scaled = X_scaled
        self.Y_scaled = Y_scaled
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

        if(setType == "test"):
            with open(path +'y_test_final_'+set_name+'.p', 'rb') as f:
                self.Y_final_ori = pickle.load(f)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def getInterval(self, i):
        return self.interval[i]
    
    def addDerevative(self, X):
        for i in range(X.shape[2]):
            X = np.append(X, librosa.feature.delta(X[:,:,i], order=1).reshape(X.shape[0], X.shape[1], -1), axis=2)
            X = np.append(X, librosa.feature.delta(X[:,:,i], order=2).reshape(X.shape[0], X.shape[1], -1), axis=2)
        return X
    
    def addDerevativeToOneRow(self, X):
        for i in range(X.shape[1]):
            X = np.append(X, librosa.feature.delta(X[:,i], order=1).reshape(X.shape[0], -1), axis=1)
            X = np.append(X, librosa.feature.delta(X[:,i], order=2).reshape(X.shape[0], -1), axis=1)
        return X


class TrainSet(Set):

    def __init__(self):
        super(TrainSet, self).__init__("train")

    def scaling(self, flag):
        if flag:
            self.X = self.X_scaled
            self.Y = self.Y_scaled
        else:
            self.X = self.X_ori
            self.Y = self.Y_ori

    def scale_x(self, x):
        assert len(x.shape) == 2, "shape of y must be (t, dim)"
        return self.x_scaler.transform(x)

    def rescale_y(self, y):
        assert len(y.shape) == 2, "shape of y must be (t, dim)"
        return self.y_scaler.inverse_transform(y)


class TestSet(Set):

    def __init__(self):
        super(TestSet, self).__init__("test")



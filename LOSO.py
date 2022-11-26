import tensorflow.compat.v1 as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.io import loadmat
import os
from pywt import wavedec
from functools import reduce
from scipy import signal
from scipy.stats import entropy
from scipy.fft import fft, ifft
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow import keras as K
import matplotlib.pyplot as plt
import scipy
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,cross_validate
from tensorflow.keras.layers import Dense, Activation, Flatten, concatenate, Input, Dropout, LSTM, Bidirectional,BatchNormalization,PReLU,ReLU,Reshape,LeakyReLU
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, Model, load_model
import matplotlib.pyplot as plt;
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from tensorflow import keras
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import Conv1D,Conv2D,Add
from tensorflow.keras.layers import MaxPool1D, MaxPooling2D
import seaborn as sns

from braindecode.models.deep4 import Deep4Net
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.util import set_random_seeds
import torch.nn.functional as F
from os.path import join as pjoin
import argparse
import json
import logging
import sys
import torch

parser = argparse.ArgumentParser(
    description='Subject-Independent EEG Confusion Classification')
parser.add_argument('-subj', type=int,
                    help='Target Subject', required=True)
args = parser.parse_args()

targ_subj = args.subj

data = pd.read_csv("./input/EEG_data.csv")
print(data.info())

demo_data = pd.read_csv('./input/demographic_info.csv')
print(demo_data)

demo_data = demo_data.rename(columns = {'subject ID': 'SubjectID'})
data = data.merge(demo_data,how = 'inner',on = 'SubjectID')
print(data.head())

data = pd.get_dummies(data)

# import pandas_profiling as pp
# pp.ProfileReport(data)

# plt.figure(figsize = (15,15))
# plt.show()
# cor_matrix = data.corr()
# sns.heatmap(cor_matrix,annot=True)
# plt.show()

# Pre-Processing
subj_marker = []
subj_marker.append(0)
subjs = data.pop('SubjectID')
current_subj = 1
for i in range(0,len(subjs)):
    if subjs[i] == current_subj:
        subj_marker.append(i-1)
        current_subj += 1
subj_marker.append(len(subjs)-1)
print(subj_marker)

data.drop(columns = ['VideoID','predefinedlabel','Attention','Mediation'],inplace=True)
y= data.pop('user-definedlabeln')
print(y.shape)
for i in range(0,len(y)):
    if y[i] <= 0.0:
        y[i] = 0.0
    else:
        y[i] = 1.0
x= data
print(np.unique(y))
# x.iloc[:1000,:11].plot(figsize = (15,10))
x = StandardScaler().fit_transform(x)
# pd.DataFrame(x).iloc[:1000,:11].plot(figsize = (15,10))

# Data Split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)

print(x.shape)
subj_start = subj_marker[targ_subj-1]
subj_end = subj_marker[targ_subj]
x_test = x[subj_start:subj_end,:]
y_test = y[subj_start:subj_end]
if targ_subj > 1:
    x_train = np.concatenate((x[:subj_start,:], x[subj_end:,:]),axis=0)
    y_train = np.concatenate((y[:subj_start], y[subj_end:]),axis=0)
else:
    x_train = x[subj_end:,:]
    y_train = y[subj_end:]

x_train = np.array(x_train).reshape(-1,15,1)
x_test = np.array(x_test).reshape(-1,15,1)

# x_train = x_train.astype(np.float32)
# x_test = x_test.astype(np.float32)
# x_train = torch.from_numpy(x_train)
# x_test = torch.from_numpy(x_test)
# x_train = x_train[:,:,np.newaxis,:].to('cuda')
# x_test = x_test[:,:,np.newaxis,:].to('cuda')

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

# Model
filters = 8

inputs = tf.keras.Input(shape=(15,1))
# inputs = Input(shape=x_train.shape[1:], name='encoder_input')

Dense1 = Dense(256, activation = 'relu',kernel_regularizer=keras.regularizers.l2())(inputs)

# x = Conv2D(filters=filters,kernel_size=(1, 50),strides=(1,25),)(Dense1)
# x = BatchNormalization()(x)
# x = LeakyReLU(alpha=0.2)(x)

# Dense2 = Dense(512, activation = 'relu',kernel_regularizer=keras.regularizers.l2())(Dense1)
# Dense3 = Dense(256, activation = 'relu',kernel_regularizer=keras.regularizers.l2())(Dense2)

lstm_1=  Bidirectional(LSTM(256, return_sequences = True))(Dense1)
drop = Dropout(0.4)(lstm_1)
lstm_3=  Bidirectional(LSTM(128, return_sequences = True))(drop)
drop2 = Dropout(0.4)(lstm_3)

flat = Flatten()(drop2)

Dense_1 = Dense(128, activation = 'relu')(flat)

# Dense_2 = Dense(128, activation = 'relu')(Dense_1)
outputs = Dense(1, activation='sigmoid')(Dense_1)

model = tf.keras.Model(inputs, outputs)

model.summary()

# tf.keras.utils.plot_model(model)

def train_model(model,x_train, y_train,x_test,y_test, save_to, epoch = 2):

        opt_adam = keras.optimizers.Adam(learning_rate=0.01)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint(save_to + '_best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * np.exp(-epoch / 10.))
        
        model.compile(optimizer=opt_adam,
                  loss=['binary_crossentropy'],
                  metrics=['accuracy'])
        
        history = model.fit(x_train,y_train,
                        batch_size=20,
                        epochs=epoch,
                        validation_data=(x_test,y_test),
                        callbacks=[es,mc,lr_schedule])
        
        saved_model = load_model(save_to + '_best_model.h5')
        
        return model,history

model,history = train_model(model, x_train, y_train,x_test, y_test, save_to= './', epoch = 50) 

# Plotting Results
# plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_pred =model.predict(x_test)
y_pred = np.array(y_pred >= 0.5, dtype = np.int)
confusion_matrix(y_test, y_pred)
print(y_pred.shape)

print(classification_report(y_test, y_pred))
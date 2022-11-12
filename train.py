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
from tensorflow.keras.layers import Dense, Activation, Flatten, concatenate, Input, Dropout, LSTM, Bidirectional,BatchNormalization,PReLU,ReLU,Reshape
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
data.drop(columns = ['SubjectID','VideoID','predefinedlabel','user-definedlabeln','Mediation'],inplace=True)
y= data.pop('Attention')
print(y.shape)
for i in range(0,len(y)):
    if y[i] <= 50.0:
        y[i] = 0.0
    else:
        y[i] = 1.0
x= data
# x.iloc[:1000,:11].plot(figsize = (15,10))
x = StandardScaler().fit_transform(x)
# pd.DataFrame(x).iloc[:1000,:11].plot(figsize = (15,10))

# Data Split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)
x_train.shape, x_test.shape,y_train.shape,y_test.shape
x_train = np.array(x_train).reshape(-1,15,1)
x_test = np.array(x_test).reshape(-1,15,1)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

# Model

inputs = tf.keras.Input(shape=(15,1))

Dense1 = Dense(64, activation = 'relu',kernel_regularizer=keras.regularizers.l2())(inputs)

#Dense2 = Dense(128, activation = 'relu',kernel_regularizer=keras.regularizers.l2())(Dense1)
#Dense3 = Dense(256, activation = 'relu',kernel_regularizer=keras.regularizers.l2())(Dense2)

lstm_1=  Bidirectional(LSTM(256, return_sequences = True))(Dense1)
drop = Dropout(0.3)(lstm_1)
lstm_3=  Bidirectional(LSTM(128, return_sequences = True))(drop)
drop2 = Dropout(0.3)(lstm_3)

flat = Flatten()(drop2)

#Dense_1 = Dense(256, activation = 'relu')(flat)

Dense_2 = Dense(128, activation = 'relu')(flat)
outputs = Dense(1, activation='sigmoid')(Dense_2)

model = tf.keras.Model(inputs, outputs)

model.summary()

# tf.keras.utils.plot_model(model)

def train_model(model,x_train, y_train,x_test,y_test, save_to, epoch = 2):

        opt_adam = keras.optimizers.Adam(learning_rate=0.001)

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

model,history = train_model(model, x_train, y_train,x_test, y_test, save_to= './', epoch = 100) 

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

# X_train = x_train.T
# X_test = x_test.T
# X_train = X_train.astype(np.float32)
# X_test = X_test.astype(np.float32)
# X_train = torch.from_numpy(X_train)
# X_test = torch.from_numpy(X_test)
# X_train = X_train[:,np.newaxis,:,:].to('cuda')
# X_test = X_test[:,np.newaxis,:,:].to('cuda')


# Y_train = y_train.astype(np.int64)
# Y_test = y_test.astype(np.int64)

# print(X_train.shape,Y_train.shape)

# train_set = SignalAndTarget(X_train, y=Y_train)
# valid_set = SignalAndTarget(X_train, y=Y_train)
# # valid_set = SignalAndTarget(new_X_val[:,0,:,:], y=Y_val)
# test_set = SignalAndTarget(X_test, y=Y_test)
# n_classes = 4
# in_chans = train_set.X.shape[1]

# model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
#                     input_time_length=train_set.X.shape[2],
#                     final_conv_length='auto').cuda()

# optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001)
# model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, )


# exp = model.fit(train_set.X, train_set.y, epochs=200, batch_size=1, scheduler='cosine',
#                     validation_data=(valid_set.X, valid_set.y), remember_best_column='valid_loss')
# rememberer = exp.rememberer
# base_model_param = {
#     'epoch': rememberer.best_epoch,
#     'model_state_dict': rememberer.model_state_dict,
#     'optimizer_state_dict': rememberer.optimizer_state_dict,
#     'loss': rememberer.lowest_val
# }
# torch.save(base_model_param, pjoin(
#     './results/', 'DG_model_subj{}.pt'.format(1)))
# model.epochs_df.to_csv(
#     pjoin('./results/', 'DG_epochs_subj{}.csv'.format(1)))


# test_loss = model.evaluate(test_set.X, test_set.y)
# print(test_loss)
# with open(pjoin('./results/', 'test_base_subj{}.json'.format(1)), 'w') as f:
#     json.dump(test_loss, f)
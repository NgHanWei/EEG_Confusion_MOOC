import tensorflow.compat.v1 as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import seaborn as sns

from braindecode.models.deep4 import Deep4Net
from braindecode.models.eegnet import EEGNetv4
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
    description='Subject and Video-Independent EEG Confusion Classification')
parser.add_argument('--normalize',default=False, help='Normalize Data', action='store_true')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)
torch.cuda.set_device(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
set_random_seeds(seed=2022, cuda=True)
BATCH_SIZE = 16
TRAIN_EPOCH = 100

normalize = args.normalize
show = True

accuracy_array = np.zeros((10, 10))
for t_vid in range(1,11):
    for t_subj in range(1,11):


        targ_vid = t_vid
        targ_subj = t_subj

        data = pd.read_csv("./input/EEG_data.csv")
        print(data.info())

        demo_data = pd.read_csv('./input/demographic_info.csv')
        print(demo_data)

        demo_data = demo_data.rename(columns = {'subject ID': 'SubjectID'})
        data = data.merge(demo_data,how = 'inner',on = 'SubjectID')
        print(data.head())

        data = pd.get_dummies(data)
        
        if show == True:
            sns.set(font_scale=0.7)
            cor_matrix = data.corr()
            mask = np.triu(np.ones_like(cor_matrix, dtype=np.bool))
            plt.figure(figsize=(16, 6))
            heatmap = sns.heatmap(cor_matrix,mask=mask,vmin=-1, vmax=1,cmap='BrBG',square = True, annot=True, annot_kws={"fontsize":5})
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
            # heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
            plt.show()
            show = False

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

        vid_marker = []
        vids = data.pop('VideoID')
        current_subj = 0
        for i in range(0,len(vids)):
            if vids[i] != current_subj:
                vid_marker.append(i)
                current_subj = vids[i]
        vid_marker.append(len(vids))
        print(vid_marker)
        print(len(vid_marker))

        data.drop(columns = ['Predefined_Label','Attention','Mediation'],inplace=True)
        y= data.pop('User-defined_Label')
        print(y.shape)
        for i in range(0,len(y)):
            if y[i] <= 0.0:
                y[i] = 0.0
            else:
                y[i] = 1.0
        x= data

        x = StandardScaler().fit_transform(x)

        ## Data Split
        # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)

        ## Subject Split
        print(x.shape)
        # subj_start = subj_marker[targ_subj-1]
        # subj_end = subj_marker[targ_subj]
        # x_test = x[subj_start:subj_end,:]
        # y_test = y[subj_start:subj_end]
        # if targ_subj > 1:
        #     x_train = np.concatenate((x[:subj_start,:], x[subj_end:,:]),axis=0)
        #     y_train = np.concatenate((y[:subj_start], y[subj_end:]),axis=0)
        # else:
        #     x_train = x[subj_end:,:]
        #     y_train = y[subj_end:]

        ## Video Split
        sliding_window = 100
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for i in range(0,len(vid_marker)):

            # Get one Video
            if i == 0:
                x_vid = x[:vid_marker[i]]
                y_vid = y[:vid_marker[i]]
            else:
                vid_start = vid_marker[i-1]
                vid_end = vid_marker[i]
                x_vid = x[vid_start:vid_end]
                y_vid = y[vid_start:vid_end]

            print(y_vid)
            # Process Video, split one video into many smaller chunks, all taking same label
            steps = len(x_vid) - sliding_window
            x_vid_concat = []
            y_vid_concat = []
            for j in range(0,steps+1):
                x_vid_slice = x_vid[j:sliding_window+j].T

                # Normalization
                if normalize == True:
                    row_sums = x_vid_slice.sum(axis=1)
                    x_vid_slice = x_vid_slice / row_sums[:, np.newaxis]

                x_vid_slice = x_vid_slice[np.newaxis,:,:].astype(np.float32)

                # print(x_vid_slice.shape)

                x_vid_concat = np.concatenate((x_vid_concat, x_vid_slice),axis=0) if len(x_vid_concat) >= 1 else x_vid_slice
                y_vid_concat = np.concatenate((y_vid_concat, [y_vid[vid_marker[i]-1].astype(np.int64)]),axis=0) if len(y_vid_concat) >= 1 else [y_vid[vid_marker[i]-1].astype(np.int64)]
            
            print(x_vid_concat.shape)
            print(y_vid_concat)

            # Target Subject Data do not append to train
            if ((i+1)% 10 == targ_vid) or ((i+1)%10 == 0 and targ_vid == 10):
                if i <= (targ_subj * 10 - 1) and i >= (targ_subj * 10 - 10):
                    x_test = np.concatenate((x_test, x_vid_concat),axis=0) if len(x_test) >= 1 else x_vid_concat
                    y_test = np.concatenate((y_test, y_vid_concat),axis=0) if len(y_test) >= 1 else y_vid_concat
                    print("Skip")
            elif i > (targ_subj * 10 - 1) or i < (targ_subj * 10 - 10):
                x_train = np.concatenate((x_train, x_vid_concat),axis=0) if len(x_train) >= 1 else x_vid_concat
                y_train = np.concatenate((y_train, y_vid_concat),axis=0) if len(y_train) >= 1 else y_vid_concat

        while x_train.shape[2] < 800:
            x_train = np.concatenate((x_train,x_train),axis = 2)
            x_test = np.concatenate((x_test,x_test),axis = 2)

        print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
        X_train = x_train
        X_test = x_test
        # X_train = X_train.astype(np.float32)
        # X_test = X_test.astype(np.float32)
        # X_train = torch.from_numpy(X_train)
        # X_test = torch.from_numpy(X_test)
        # X_train = X_train.to('cuda')
        # X_test = X_test.to('cuda')

        Y_train = y_train
        Y_test = y_test

        train_set = SignalAndTarget(X_train, y=Y_train)
        valid_set = SignalAndTarget(X_test, y=Y_test)
        test_set = SignalAndTarget(X_test, y=Y_test)
        n_classes = 2
        in_chans = train_set.X.shape[1]

        model = EEGNetv4(in_chans=in_chans, n_classes=n_classes,
                            input_time_length=train_set.X.shape[2],
                            final_conv_length='auto').cuda()

        optimizer = AdamW(model.parameters(), lr=1*0.001, weight_decay=0.5*0.001)
        model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, )


        exp = model.fit(train_set.X, train_set.y, epochs=TRAIN_EPOCH,
                        batch_size=BATCH_SIZE, scheduler='cosine',
                        validation_data=(valid_set.X, valid_set.y), remember_best_column='valid_loss')
        rememberer = exp.rememberer
        base_model_param = {
            'epoch': rememberer.best_epoch,
            'model_state_dict': rememberer.model_state_dict,
            'optimizer_state_dict': rememberer.optimizer_state_dict,
            'loss': rememberer.lowest_val
        }
        torch.save(base_model_param, pjoin(
            './results/', 'model_subj{}.pt'.format(targ_vid)))
        model.epochs_df.to_csv(
            pjoin('./results/', 'epochs_subj{}.csv'.format(targ_vid)))


        test_loss = model.evaluate(test_set.X, test_set.y)
        print(test_loss)
        with open(pjoin('./results/', 'test_base_subj{}.json'.format(targ_vid)), 'w') as f:
            json.dump(test_loss, f)

        accuracy = (1-test_loss['misclass']) * 100
        accuracy_array[t_vid-1,t_subj-1] = int(accuracy)


        np.savetxt("firstarray.csv", accuracy_array, delimiter=",")
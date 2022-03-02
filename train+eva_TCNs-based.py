# -----------------------------------------------------------------------------
# written by Jeongho Ahn
# Data: 2022/02/10
# -----------------------------------------------------------------------------

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import os
import sys
import re
from typing import List, Dict, Tuple, Set
import csv
import pathlib
import itertools

import keras
import tensorflow as tf
#import tensorflow_addons as tfa
import tcn
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Concatenate, Lambda, Reshape
from keras.layers.core import Dense, Activation, Dropout, Flatten
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from tensorflow.keras import optimizers

from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# -----------------------------------------------------------------------------

learning_mode = False
evaluation_mode = True
prediction_mode = False

element20to10_mode = True

class_num = 2
database_20210820_26_1 = False
database_20210820_26_2 = False
database_20210820_26_3 = False
database_20210820_26_all = True
pattern_1 = True
pattern_2 = False
pattern_3 = False

tcn_baseline_1 = False
tcn_baseline_2 = True
tcn_baseline_3 = False
lstm_baseline = False

scaling_1div100 = True
timeseries_num = 25
time_threshold = 15
time_max = 40
rate = 0.8

batch_size = 200
epochs = 100
earlystopping = True
patience = 50

# -----------------------------------------------------------------------------

if database_20210820_26_1:
    csv_files = glob.glob('./database_20210820-26/database1/*Combine.csv')
elif database_20210820_26_2:
    csv_files = glob.glob('./database_20210820-26/database2/*Combine.csv')
elif database_20210820_26_3:
    csv_files = glob.glob('./database_20210820-26/database3/*Combine.csv')
elif database_20210820_26_all:
    csv_files = glob.glob('./database_20211002/all/*Combine.csv')
print('\nlen(csv_files) = ', len(csv_files))
    
x_train = None
x_val = None
x_test = None
y_train = None
y_val = None
y_test = None

csv_column = list(range(1, 61, 3))
for csv_file in csv_files:
    csv_file_name = os.path.basename(csv_file)
    print('\ncsv_file_name = ', csv_file_name)
    with open(csv_file, 'r', encoding='shift_jis') as f:
        reader = list(csv.reader(f))
        print('len(reader)_before = ', len(reader))
        #reader = reader[(int(8)+int(time_threshold/rate)):(int(8)+int(time_max/rate))] # 13, 18, 23, 28, 33 # maru
        reader = reader[(int(7)+int(time_threshold/rate)):(int(7)+int(time_max/rate))] # 13, 18, 23, 28, 33
        print('len(reader)_after = ', len(reader))
        reader = np.array(reader)
        print('reader.shape = ', reader.shape)
        reader = reader[:,csv_column].astype(np.float64)
        if element20to10_mode:
            element_20CK = reader[:,0:1]
            element_20CN = reader[:,3:4]
            element_20CP = reader[:,5:6]
            element_20CQ = reader[:,6:7]
            element_20CR = reader[:,7:8]
            element_20CT = reader[:,9:10]
            element_20CX = reader[:,13:14]
            element_20CY = reader[:,14:15]
            element_20DB = reader[:,17:18]
            element_20DC = reader[:,18:19]
            reader = np.concatenate([element_20CY, element_20CN, element_20CR, element_20CT, element_20CK, 
                                     element_20DC, element_20CX, element_20DB, element_20CP, element_20CQ], axis=-1)
        print('reader.shape = ', reader.shape)
        
# -----------------------------------------------------------------------------

        if pattern_1:
            if (csv_file_name == '20210922101306_coffee_Combine.csv'
                  or csv_file_name == '20210922101405_coffee_Combine.csv'
                  or csv_file_name == '20210922101506_coffee_Combine.csv'
                  or csv_file_name == '20210922101605_coffee_Combine.csv'
                  or csv_file_name == '20210922101704_coffee_Combine.csv'
                  or csv_file_name == '20210922101805_coffee_Combine.csv'
                  or csv_file_name == '20210922101904_coffee_Combine.csv'
                  or csv_file_name == '20210922102005_coffee_Combine.csv'
                  or csv_file_name == '20210922102105_coffee_Combine.csv'
                  or csv_file_name == '20210922102204_coffee_Combine.csv'
                  or csv_file_name == '20210922114605_coffee_Combine.csv'
                  or csv_file_name == '20210922114704_coffee_Combine.csv'
                  or csv_file_name == '20210922114805_coffee_Combine.csv'
                  or csv_file_name == '20210922114904_coffee_Combine.csv'
                  or csv_file_name == '20210922115005_coffee_Combine.csv'
                  or csv_file_name == '20210922115104_coffee_Combine.csv'
                  or csv_file_name == '20210922115205_coffee_Combine.csv'
                  or csv_file_name == '20210922115304_coffee_Combine.csv'
                  or csv_file_name == '20210922115404_coffee_Combine.csv'
                  or csv_file_name == '20210922115505_coffee_Combine.csv'
                  or csv_file_name == '20210922115605_coffee_Combine.csv'
                  or csv_file_name == '20210922115705_coffee_Combine.csv'
                  or csv_file_name == '20210922115805_coffee_Combine.csv'
                  or csv_file_name == '20210922115905_coffee_Combine.csv'
                  or csv_file_name == '20210922120005_coffee_Combine.csv'
                  or csv_file_name == '20210922120105_coffee_Combine.csv'
                  or csv_file_name == '20210922120205_coffee_Combine.csv'
                  or csv_file_name == '20210922120305_coffee_Combine.csv'
                  or csv_file_name == '20210922120405_coffee_Combine.csv'
                  or csv_file_name == '20210922120505_coffee_Combine.csv'
                  or csv_file_name == '20210922133505_coffee_Combine.csv'
                  or csv_file_name == '20210922133605_coffee_Combine.csv'
                  or csv_file_name == '20210922133705_coffee_Combine.csv'
                  or csv_file_name == '20210922133805_coffee_Combine.csv'
                  or csv_file_name == '20210922133905_coffee_Combine.csv'
                  or csv_file_name == '20210922134005_coffee_Combine.csv'
                  or csv_file_name == '20210922134105_coffee_Combine.csv'
                  or csv_file_name == '20210922134205_coffee_Combine.csv'
                  or csv_file_name == '20210922134305_coffee_Combine.csv'
                  or csv_file_name == '20210922134405_coffee_Combine.csv'
                  or csv_file_name == '20210922142506_coffee_Combine.csv'
                  or csv_file_name == '20210922142605_coffee_Combine.csv'
                  or csv_file_name == '20210922142705_coffee_Combine.csv'
                  or csv_file_name == '20210922142806_coffee_Combine.csv'
                  or csv_file_name == '20210922142905_coffee_Combine.csv'
                  or csv_file_name == '20210922143005_coffee_Combine.csv'
                  or csv_file_name == '20210922143105_coffee_Combine.csv'
                  or csv_file_name == '20210922143211_coffee_Combine.csv'
                  or csv_file_name == '20210922143313_coffee_Combine.csv'
                  or csv_file_name == '20210922143408_coffee_Combine.csv'
                  or csv_file_name == '20210922160205_coffee_Combine.csv'
                  or csv_file_name == '20210922160312_coffee_Combine.csv'
                  or csv_file_name == '20210922160405_coffee_Combine.csv'
                  or csv_file_name == '20210922160506_coffee_Combine.csv'
                  or csv_file_name == '20210922160606_coffee_Combine.csv'
                  or csv_file_name == '20210922160705_coffee_Combine.csv'
                  or csv_file_name == '20210922160805_coffee_Combine.csv'
                  or csv_file_name == '20210922160906_coffee_Combine.csv'
                  or csv_file_name == '20210922161005_coffee_Combine.csv'
                  or csv_file_name == '20210922161106_coffee_Combine.csv'
                  or csv_file_name == '20210924152006_coffee_Combine.csv'
                  or csv_file_name == '20210924152105_coffee_Combine.csv'
                  or csv_file_name == '20210924152206_coffee_Combine.csv'
                  or csv_file_name == '20210924152306_coffee_Combine.csv'
                  or csv_file_name == '20210924152405_coffee_Combine.csv'
                  or csv_file_name == '20210924152506_coffee_Combine.csv'
                  or csv_file_name == '20210924152605_coffee_Combine.csv'
                  or csv_file_name == '20210924152705_coffee_Combine.csv'
                  or csv_file_name == '20210924152805_coffee_Combine.csv'
                  or csv_file_name == '20210924152905_coffee_Combine.csv'
                  or csv_file_name == '20210924171605_coffee_Combine.csv'
                  or csv_file_name == '20210924171704_coffee_Combine.csv'
                  or csv_file_name == '20210924171805_coffee_Combine.csv'
                  or csv_file_name == '20210924171908_coffee_Combine.csv'
                  or csv_file_name == '20210924172012_coffee_Combine.csv'
                  or csv_file_name == '20210924172104_coffee_Combine.csv'
                  or csv_file_name == '20210924172206_coffee_Combine.csv'
                  or csv_file_name == '20210924172305_coffee_Combine.csv'
                  or csv_file_name == '20210924172405_coffee_Combine.csv'
                  or csv_file_name == '20210924172504_coffee_Combine.csv'
                  or csv_file_name == '20210924173304_coffee_Combine.csv'
                  or csv_file_name == '20210924173404_coffee_Combine.csv'
                  or csv_file_name == '20210924173505_coffee_Combine.csv'
                  or csv_file_name == '20210924173605_coffee_Combine.csv'
                  or csv_file_name == '20210924173704_coffee_Combine.csv'
                  or csv_file_name == '20210924173805_coffee_Combine.csv'
                  or csv_file_name == '20210924173904_coffee_Combine.csv'
                  or csv_file_name == '20210924174005_coffee_Combine.csv'
                  or csv_file_name == '20210924174105_coffee_Combine.csv'
                  or csv_file_name == '20210924174205_coffee_Combine.csv'
                  or csv_file_name == '20210924191005_coffee_Combine.csv'
                  or csv_file_name == '20210924191104_coffee_Combine.csv'
                  or csv_file_name == '20210924191204_coffee_Combine.csv'
                  or csv_file_name == '20210924191304_coffee_Combine.csv'
                  or csv_file_name == '20210924191404_coffee_Combine.csv'
                  or csv_file_name == '20210924191504_coffee_Combine.csv'
                  or csv_file_name == '20210924191604_coffee_Combine.csv'
                  or csv_file_name == '20210924191704_coffee_Combine.csv'
                  or csv_file_name == '20210924191804_coffee_Combine.csv'
                  or csv_file_name == '20210924191904_coffee_Combine.csv'
                  or csv_file_name == '20210927084903_coffee_Combine.csv'
                  or csv_file_name == '20210927085003_coffee_Combine.csv'
                  or csv_file_name == '20210927085103_coffee_Combine.csv'
                  or csv_file_name == '20210927085204_coffee_Combine.csv'
                  or csv_file_name == '20210927085307_coffee_Combine.csv'
                  or csv_file_name == '20210927085404_coffee_Combine.csv'
                  or csv_file_name == '20210927085509_coffee_Combine.csv'
                  or csv_file_name == '20210927085610_coffee_Combine.csv'
                  or csv_file_name == '20210927085715_coffee_Combine.csv'
                  or csv_file_name == '20210927085803_coffee_Combine.csv'
                  or csv_file_name == '20210927090904_coffee_Combine.csv'
                  or csv_file_name == '20210927091004_coffee_Combine.csv'
                  or csv_file_name == '20210927091103_coffee_Combine.csv'
                  or csv_file_name == '20210927091203_coffee_Combine.csv'
                  or csv_file_name == '20210927091304_coffee_Combine.csv'
                  or csv_file_name == '20210927091404_coffee_Combine.csv'
                  or csv_file_name == '20210927091504_coffee_Combine.csv'
                  or csv_file_name == '20210927091604_coffee_Combine.csv'
                  or csv_file_name == '20210927091703_coffee_Combine.csv'
                  or csv_file_name == '20210927091804_coffee_Combine.csv'
                  or csv_file_name == '20210927125803_coffee_Combine.csv'
                  or csv_file_name == '20210927125903_coffee_Combine.csv'
                  or csv_file_name == '20210927130004_coffee_Combine.csv'
                  or csv_file_name == '20210927130104_coffee_Combine.csv'
                  or csv_file_name == '20210927130203_coffee_Combine.csv'
                  or csv_file_name == '20210927130303_coffee_Combine.csv'
                  or csv_file_name == '20210927130403_coffee_Combine.csv'
                  or csv_file_name == '20210927130503_coffee_Combine.csv'
                  or csv_file_name == '20210927130604_coffee_Combine.csv'
                  or csv_file_name == '20210927130704_coffee_Combine.csv'
                  or csv_file_name == '20210927131804_coffee_Combine.csv'
                  or csv_file_name == '20210927131904_coffee_Combine.csv'
                  or csv_file_name == '20210927132003_coffee_Combine.csv'
                  or csv_file_name == '20210927132108_coffee_Combine.csv'
                  or csv_file_name == '20210927132203_coffee_Combine.csv'
                  or csv_file_name == '20210927132303_coffee_Combine.csv'
                  or csv_file_name == '20210927132409_coffee_Combine.csv'
                  or csv_file_name == '20210927132507_coffee_Combine.csv'
                  or csv_file_name == '20210927132606_coffee_Combine.csv'
                  or csv_file_name == '20210927132707_coffee_Combine.csv'
                  or csv_file_name == '20210927135104_coffee_Combine.csv'
                  or csv_file_name == '20210927135204_coffee_Combine.csv'
                  or csv_file_name == '20210927135303_coffee_Combine.csv'
                  or csv_file_name == '20210927135403_coffee_Combine.csv'
                  or csv_file_name == '20210927135503_coffee_Combine.csv'
                  or csv_file_name == '20210927135603_coffee_Combine.csv'
                  or csv_file_name == '20210927135703_coffee_Combine.csv'
                  or csv_file_name == '20210927135803_coffee_Combine.csv'
                  or csv_file_name == '20210927135904_coffee_Combine.csv'
                  or csv_file_name == '20210927140003_coffee_Combine.csv'
                  or csv_file_name == '20210927172703_coffee_Combine.csv'
                  or csv_file_name == '20210927172804_coffee_Combine.csv'
                  or csv_file_name == '20210927172904_coffee_Combine.csv'
                  or csv_file_name == '20210927173003_coffee_Combine.csv'
                  or csv_file_name == '20210927173104_coffee_Combine.csv'
                  or csv_file_name == '20210927173204_coffee_Combine.csv'
                  or csv_file_name == '20210927173303_coffee_Combine.csv'
                  or csv_file_name == '20210927173404_coffee_Combine.csv'
                  or csv_file_name == '20210927173504_coffee_Combine.csv'
                  or csv_file_name == '20210927173603_coffee_Combine.csv'
                  or csv_file_name == '20210927183805_coffee_Combine.csv'
                  or csv_file_name == '20210927183904_coffee_Combine.csv'
                  or csv_file_name == '20210927184003_coffee_Combine.csv'
                  or csv_file_name == '20210927184104_coffee_Combine.csv'
                  or csv_file_name == '20210927184203_coffee_Combine.csv'
                  or csv_file_name == '20210927184304_coffee_Combine.csv'
                  or csv_file_name == '20210927184419_coffee_Combine.csv'
                  or csv_file_name == '20210927184504_coffee_Combine.csv'
                  or csv_file_name == '20210927184604_coffee_Combine.csv'
                  or csv_file_name == '20210927184704_coffee_Combine.csv'
                  or csv_file_name == '20210927201903_coffee_Combine.csv'
                  or csv_file_name == '20210927202004_coffee_Combine.csv'
                  or csv_file_name == '20210927202104_coffee_Combine.csv'
                  or csv_file_name == '20210927202204_coffee_Combine.csv'
                  or csv_file_name == '20210927202304_coffee_Combine.csv'
                  or csv_file_name == '20210927202404_coffee_Combine.csv'
                  or csv_file_name == '20210927202504_coffee_Combine.csv'
                  or csv_file_name == '20210927202614_coffee_Combine.csv'
                  or csv_file_name == '20210927202704_coffee_Combine.csv'
                  or csv_file_name == '20210927202804_coffee_Combine.csv'):
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32') #(18,15,10), reader(32,10)
                for j in range(reader.shape[0]-timeseries_num+1): #range(18)
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1]) #(1,15,10)
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 0, dtype='int16') #(18, 1) value = 0
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_val is None:
                    x_val = x_timeseries
                    y_val = y_data
                else:
                    x_val = np.concatenate((x_val, x_timeseries), axis=0)
                    y_val = np.concatenate((y_val, y_data), axis=0)
                    
            elif (csv_file_name == '20210922102304_coffee_Combine.csv'
                  or csv_file_name == '20210922102404_coffee_Combine.csv'
                  or csv_file_name == '20210922102504_coffee_Combine.csv'
                  or csv_file_name == '20210922102604_coffee_Combine.csv'
                  or csv_file_name == '20210922102706_coffee_Combine.csv'
                  or csv_file_name == '20210922102806_coffee_Combine.csv'
                  or csv_file_name == '20210922102905_coffee_Combine.csv'
                  or csv_file_name == '20210922103005_coffee_Combine.csv'
                  or csv_file_name == '20210922103104_coffee_Combine.csv'
                  or csv_file_name == '20210922103205_coffee_Combine.csv'
                  or csv_file_name == '20210922113605_coffee_Combine.csv'
                  or csv_file_name == '20210922113704_coffee_Combine.csv'
                  or csv_file_name == '20210922113806_coffee_Combine.csv'
                  or csv_file_name == '20210922113905_coffee_Combine.csv'
                  or csv_file_name == '20210922114004_coffee_Combine.csv'
                  or csv_file_name == '20210922114106_coffee_Combine.csv'
                  or csv_file_name == '20210922114205_coffee_Combine.csv'
                  or csv_file_name == '20210922114307_coffee_Combine.csv'
                  or csv_file_name == '20210922114404_coffee_Combine.csv'
                  or csv_file_name == '20210922114504_coffee_Combine.csv'
                  or csv_file_name == '20210922120605_coffee_Combine.csv'
                  or csv_file_name == '20210922120705_coffee_Combine.csv'
                  or csv_file_name == '20210922120805_coffee_Combine.csv'
                  or csv_file_name == '20210922120905_coffee_Combine.csv'
                  or csv_file_name == '20210922121005_coffee_Combine.csv'
                  or csv_file_name == '20210922121105_coffee_Combine.csv'
                  or csv_file_name == '20210922121205_coffee_Combine.csv'
                  or csv_file_name == '20210922121305_coffee_Combine.csv'
                  or csv_file_name == '20210922121405_coffee_Combine.csv'
                  or csv_file_name == '20210922121505_coffee_Combine.csv'
                  or csv_file_name == '20210922132506_coffee_Combine.csv'
                  or csv_file_name == '20210922132605_coffee_Combine.csv'
                  or csv_file_name == '20210922132705_coffee_Combine.csv'
                  or csv_file_name == '20210922132805_coffee_Combine.csv'
                  or csv_file_name == '20210922132905_coffee_Combine.csv'
                  or csv_file_name == '20210922133006_coffee_Combine.csv'
                  or csv_file_name == '20210922133105_coffee_Combine.csv'
                  or csv_file_name == '20210922133205_coffee_Combine.csv'
                  or csv_file_name == '20210922133306_coffee_Combine.csv'
                  or csv_file_name == '20210922133405_coffee_Combine.csv'
                  or csv_file_name == '20210922143505_coffee_Combine.csv'
                  or csv_file_name == '20210922143605_coffee_Combine.csv'
                  or csv_file_name == '20210922143705_coffee_Combine.csv'
                  or csv_file_name == '20210922143805_coffee_Combine.csv'
                  or csv_file_name == '20210922143906_coffee_Combine.csv'
                  or csv_file_name == '20210922144006_coffee_Combine.csv'
                  or csv_file_name == '20210922144105_coffee_Combine.csv'
                  or csv_file_name == '20210922144205_coffee_Combine.csv'
                  or csv_file_name == '20210922144305_coffee_Combine.csv'
                  or csv_file_name == '20210922144405_coffee_Combine.csv'
                  or csv_file_name == '20210922155205_coffee_Combine.csv'
                  or csv_file_name == '20210922155305_coffee_Combine.csv'
                  or csv_file_name == '20210922155413_coffee_Combine.csv'
                  or csv_file_name == '20210922155513_coffee_Combine.csv'
                  or csv_file_name == '20210922155612_coffee_Combine.csv'
                  or csv_file_name == '20210922155709_coffee_Combine.csv'
                  or csv_file_name == '20210922155809_coffee_Combine.csv'
                  or csv_file_name == '20210922155905_coffee_Combine.csv'
                  or csv_file_name == '20210922160006_coffee_Combine.csv'
                  or csv_file_name == '20210922160105_coffee_Combine.csv'
                  or csv_file_name == '20210924153004_coffee_Combine.csv'
                  or csv_file_name == '20210924153105_coffee_Combine.csv'
                  or csv_file_name == '20210924153210_coffee_Combine.csv'
                  or csv_file_name == '20210924153304_coffee_Combine.csv'
                  or csv_file_name == '20210924153404_coffee_Combine.csv'
                  or csv_file_name == '20210924153505_coffee_Combine.csv'
                  or csv_file_name == '20210924153604_coffee_Combine.csv'
                  or csv_file_name == '20210924153707_coffee_Combine.csv'
                  or csv_file_name == '20210924153805_coffee_Combine.csv'
                  or csv_file_name == '20210924153904_coffee_Combine.csv'
                  or csv_file_name == '20210924170611_coffee_Combine.csv'
                  or csv_file_name == '20210924170705_coffee_Combine.csv'
                  or csv_file_name == '20210924170805_coffee_Combine.csv'
                  or csv_file_name == '20210924170904_coffee_Combine.csv'
                  or csv_file_name == '20210924171004_coffee_Combine.csv'
                  or csv_file_name == '20210924171111_coffee_Combine.csv'
                  or csv_file_name == '20210924171205_coffee_Combine.csv'
                  or csv_file_name == '20210924171305_coffee_Combine.csv'
                  or csv_file_name == '20210924171404_coffee_Combine.csv'
                  or csv_file_name == '20210924171511_coffee_Combine.csv'
                  or csv_file_name == '20210924174305_coffee_Combine.csv'
                  or csv_file_name == '20210924174405_coffee_Combine.csv'
                  or csv_file_name == '20210924174505_coffee_Combine.csv'
                  or csv_file_name == '20210924174605_coffee_Combine.csv'
                  or csv_file_name == '20210924174704_coffee_Combine.csv'
                  or csv_file_name == '20210924174804_coffee_Combine.csv'
                  or csv_file_name == '20210924174905_coffee_Combine.csv'
                  or csv_file_name == '20210924175005_coffee_Combine.csv'
                  or csv_file_name == '20210924175105_coffee_Combine.csv'
                  or csv_file_name == '20210924175205_coffee_Combine.csv'
                  or csv_file_name == '20210924190004_coffee_Combine.csv'
                  or csv_file_name == '20210924190105_coffee_Combine.csv'
                  or csv_file_name == '20210924190210_coffee_Combine.csv'
                  or csv_file_name == '20210924190304_coffee_Combine.csv'
                  or csv_file_name == '20210924190404_coffee_Combine.csv'
                  or csv_file_name == '20210924190504_coffee_Combine.csv'
                  or csv_file_name == '20210924190604_coffee_Combine.csv'
                  or csv_file_name == '20210924190704_coffee_Combine.csv'
                  or csv_file_name == '20210924190805_coffee_Combine.csv'
                  or csv_file_name == '20210924190904_coffee_Combine.csv'
                  or csv_file_name == '20210927085904_coffee_Combine.csv'
                  or csv_file_name == '20210927090004_coffee_Combine.csv'
                  or csv_file_name == '20210927090103_coffee_Combine.csv'
                  or csv_file_name == '20210927090203_coffee_Combine.csv'
                  or csv_file_name == '20210927090304_coffee_Combine.csv'
                  or csv_file_name == '20210927090404_coffee_Combine.csv'
                  or csv_file_name == '20210927090503_coffee_Combine.csv'
                  or csv_file_name == '20210927090604_coffee_Combine.csv'
                  or csv_file_name == '20210927090703_coffee_Combine.csv'
                  or csv_file_name == '20210927090803_coffee_Combine.csv'
                  or csv_file_name == '20210927091904_coffee_Combine.csv'
                  or csv_file_name == '20210927092003_coffee_Combine.csv'
                  or csv_file_name == '20210927092103_coffee_Combine.csv'
                  or csv_file_name == '20210927092203_coffee_Combine.csv'
                  or csv_file_name == '20210927092305_coffee_Combine.csv'
                  or csv_file_name == '20210927092403_coffee_Combine.csv'
                  or csv_file_name == '20210927092504_coffee_Combine.csv'
                  or csv_file_name == '20210927092603_coffee_Combine.csv'
                  or csv_file_name == '20210927092704_coffee_Combine.csv'
                  or csv_file_name == '20210927092803_coffee_Combine.csv'
                  or csv_file_name == '20210927124803_coffee_Combine.csv'
                  or csv_file_name == '20210927124903_coffee_Combine.csv'
                  or csv_file_name == '20210927125003_coffee_Combine.csv'
                  or csv_file_name == '20210927125103_coffee_Combine.csv'
                  or csv_file_name == '20210927125204_coffee_Combine.csv'
                  or csv_file_name == '20210927125303_coffee_Combine.csv'
                  or csv_file_name == '20210927125404_coffee_Combine.csv'
                  or csv_file_name == '20210927125504_coffee_Combine.csv'
                  or csv_file_name == '20210927125603_coffee_Combine.csv'
                  or csv_file_name == '20210927125704_coffee_Combine.csv'
                  or csv_file_name == '20210927130803_coffee_Combine.csv'
                  or csv_file_name == '20210927130903_coffee_Combine.csv'
                  or csv_file_name == '20210927131003_coffee_Combine.csv'
                  or csv_file_name == '20210927131103_coffee_Combine.csv'
                  or csv_file_name == '20210927131204_coffee_Combine.csv'
                  or csv_file_name == '20210927131307_coffee_Combine.csv'
                  or csv_file_name == '20210927131404_coffee_Combine.csv'
                  or csv_file_name == '20210927131505_coffee_Combine.csv'
                  or csv_file_name == '20210927131604_coffee_Combine.csv'
                  or csv_file_name == '20210927131702_coffee_Combine.csv'
                  or csv_file_name == '20210927140103_coffee_Combine.csv'
                  or csv_file_name == '20210927140204_coffee_Combine.csv'
                  or csv_file_name == '20210927140304_coffee_Combine.csv'
                  or csv_file_name == '20210927140403_coffee_Combine.csv'
                  or csv_file_name == '20210927140504_coffee_Combine.csv'
                  or csv_file_name == '20210927140603_coffee_Combine.csv'
                  or csv_file_name == '20210927140704_coffee_Combine.csv'
                  or csv_file_name == '20210927140803_coffee_Combine.csv'
                  or csv_file_name == '20210927140904_coffee_Combine.csv'
                  or csv_file_name == '20210927141003_coffee_Combine.csv'
                  or csv_file_name == '20210927171704_coffee_Combine.csv'
                  or csv_file_name == '20210927171804_coffee_Combine.csv'
                  or csv_file_name == '20210927171903_coffee_Combine.csv'
                  or csv_file_name == '20210927172003_coffee_Combine.csv'
                  or csv_file_name == '20210927172103_coffee_Combine.csv'
                  or csv_file_name == '20210927172203_coffee_Combine.csv'
                  or csv_file_name == '20210927172304_coffee_Combine.csv'
                  or csv_file_name == '20210927172403_coffee_Combine.csv'
                  or csv_file_name == '20210927172503_coffee_Combine.csv'
                  or csv_file_name == '20210927172604_coffee_Combine.csv'
                  or csv_file_name == '20210927184804_coffee_Combine.csv'
                  or csv_file_name == '20210927184905_coffee_Combine.csv'
                  or csv_file_name == '20210927185003_coffee_Combine.csv'
                  or csv_file_name == '20210927185103_coffee_Combine.csv'
                  or csv_file_name == '20210927185204_coffee_Combine.csv'
                  or csv_file_name == '20210927185304_coffee_Combine.csv'
                  or csv_file_name == '20210927185404_coffee_Combine.csv'
                  or csv_file_name == '20210927185503_coffee_Combine.csv'
                  or csv_file_name == '20210927185604_coffee_Combine.csv'
                  or csv_file_name == '20210927185706_coffee_Combine.csv'
                  or csv_file_name == '20210927200906_coffee_Combine.csv'
                  or csv_file_name == '20210927201006_coffee_Combine.csv'
                  or csv_file_name == '20210927201104_coffee_Combine.csv'
                  or csv_file_name == '20210927201205_coffee_Combine.csv'
                  or csv_file_name == '20210927201304_coffee_Combine.csv'
                  or csv_file_name == '20210927201407_coffee_Combine.csv'
                  or csv_file_name == '20210927201506_coffee_Combine.csv'
                  or csv_file_name == '20210927201609_coffee_Combine.csv'
                  or csv_file_name == '20210927201704_coffee_Combine.csv'
                  or csv_file_name == '20210927201805_coffee_Combine.csv'):
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 0, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_test is None:
                    x_test = x_timeseries
                    y_test = y_data
                else:
                    x_test = np.concatenate((x_test, x_timeseries), axis=0)
                    y_test = np.concatenate((y_test, y_data), axis=0)
                    
            elif 'coffee_Combine.csv' in csv_file_name:
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 0, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_train is None:
                    x_train = x_timeseries
                    y_train = y_data
                else:
                    x_train = np.concatenate((x_train, x_timeseries), axis=0)
                    y_train = np.concatenate((y_train, y_data), axis=0)
                    
            elif (csv_file_name == '20210923140506_kurokiri_Combine.csv' 
                  or csv_file_name == '20210923140605_kurokiri_Combine.csv'
                  or csv_file_name == '20210923140704_kurokiri_Combine.csv'
                  or csv_file_name == '20210923140804_kurokiri_Combine.csv'
                  or csv_file_name == '20210923140905_kurokiri_Combine.csv'
                  or csv_file_name == '20210923141004_kurokiri_Combine.csv'
                  or csv_file_name == '20210923141104_kurokiri_Combine.csv'
                  or csv_file_name == '20210923141219_kurokiri_Combine.csv'
                  or csv_file_name == '20210923141305_kurokiri_Combine.csv'
                  or csv_file_name == '20210923141405_kurokiri_Combine.csv'
                  or csv_file_name == '20210923141504_kurokiri_Combine.csv'
                  or csv_file_name == '20210923141604_kurokiri_Combine.csv'
                  or csv_file_name == '20210923141705_kurokiri_Combine.csv'
                  or csv_file_name == '20210923141804_kurokiri_Combine.csv'
                  or csv_file_name == '20210923141904_kurokiri_Combine.csv'
                  or csv_file_name == '20210923142015_kurokiri_Combine.csv'
                  or csv_file_name == '20210923142104_kurokiri_Combine.csv'
                  or csv_file_name == '20210923142204_kurokiri_Combine.csv'
                  or csv_file_name == '20210923142304_kurokiri_Combine.csv'
                  or csv_file_name == '20210923142404_kurokiri_Combine.csv'
                  or csv_file_name == '20210923142508_kurokiri_Combine.csv'
                  or csv_file_name == '20210923142604_kurokiri_Combine.csv'
                  or csv_file_name == '20210923142704_kurokiri_Combine.csv'
                  or csv_file_name == '20210923142804_kurokiri_Combine.csv'
                  or csv_file_name == '20210923142905_kurokiri_Combine.csv'
                  or csv_file_name == '20210923143004_kurokiri_Combine.csv'
                  or csv_file_name == '20210923143104_kurokiri_Combine.csv'
                  or csv_file_name == '20210923143207_kurokiri_Combine.csv'
                  or csv_file_name == '20210923143304_kurokiri_Combine.csv'
                  or csv_file_name == '20210923143405_kurokiri_Combine.csv'
                  or csv_file_name == '20210923143504_kurokiri_Combine.csv'
                  or csv_file_name == '20210923143604_kurokiri_Combine.csv'
                  or csv_file_name == '20210923143708_kurokiri_Combine.csv'
                  or csv_file_name == '20210923143809_kurokiri_Combine.csv'
                  or csv_file_name == '20210923143905_kurokiri_Combine.csv'
                  or csv_file_name == '20210923144005_kurokiri_Combine.csv'
                  or csv_file_name == '20210923144105_kurokiri_Combine.csv'
                  or csv_file_name == '20210923144204_kurokiri_Combine.csv'
                  or csv_file_name == '20210923144304_kurokiri_Combine.csv'
                  or csv_file_name == '20210923144408_kurokiri_Combine.csv'
                  or csv_file_name == '20210923202405_kurokiri_Combine.csv'
                  or csv_file_name == '20210923202505_kurokiri_Combine.csv'
                  or csv_file_name == '20210923202604_kurokiri_Combine.csv'
                  or csv_file_name == '20210923202704_kurokiri_Combine.csv'
                  or csv_file_name == '20210923202804_kurokiri_Combine.csv'
                  or csv_file_name == '20210923202905_kurokiri_Combine.csv'
                  or csv_file_name == '20210923203004_kurokiri_Combine.csv'
                  or csv_file_name == '20210923203106_kurokiri_Combine.csv'
                  or csv_file_name == '20210923203204_kurokiri_Combine.csv'
                  or csv_file_name == '20210923203305_kurokiri_Combine.csv'
                  or csv_file_name == '20210923203404_kurokiri_Combine.csv'
                  or csv_file_name == '20210923203507_kurokiri_Combine.csv'
                  or csv_file_name == '20210923203604_kurokiri_Combine.csv'
                  or csv_file_name == '20210923203705_kurokiri_Combine.csv'
                  or csv_file_name == '20210923203805_kurokiri_Combine.csv'
                  or csv_file_name == '20210923203905_kurokiri_Combine.csv'
                  or csv_file_name == '20210923204006_kurokiri_Combine.csv'
                  or csv_file_name == '20210923204105_kurokiri_Combine.csv'
                  or csv_file_name == '20210923204204_kurokiri_Combine.csv'
                  or csv_file_name == '20210923204304_kurokiri_Combine.csv'
                  or csv_file_name == '20210923204406_kurokiri_Combine.csv'
                  or csv_file_name == '20210923204505_kurokiri_Combine.csv'
                  or csv_file_name == '20210923204604_kurokiri_Combine.csv'
                  or csv_file_name == '20210923204704_kurokiri_Combine.csv'
                  or csv_file_name == '20210923204804_kurokiri_Combine.csv'
                  or csv_file_name == '20210923204905_kurokiri_Combine.csv'
                  or csv_file_name == '20210923205004_kurokiri_Combine.csv'
                  or csv_file_name == '20210923205109_kurokiri_Combine.csv'
                  or csv_file_name == '20210923205205_kurokiri_Combine.csv'
                  or csv_file_name == '20210923205305_kurokiri_Combine.csv'
                  or csv_file_name == '20210923205405_kurokiri_Combine.csv'
                  or csv_file_name == '20210923205506_kurokiri_Combine.csv'
                  or csv_file_name == '20210923205604_kurokiri_Combine.csv'
                  or csv_file_name == '20210923205705_kurokiri_Combine.csv'
                  or csv_file_name == '20210923205805_kurokiri_Combine.csv'
                  or csv_file_name == '20210923205904_kurokiri_Combine.csv'
                  or csv_file_name == '20210923210005_kurokiri_Combine.csv'
                  or csv_file_name == '20210923210104_kurokiri_Combine.csv'
                  or csv_file_name == '20210923210204_kurokiri_Combine.csv'
                  or csv_file_name == '20210923210304_kurokiri_Combine.csv'
                  or csv_file_name == '20210926110408_kurokiri_Combine.csv'
                  or csv_file_name == '20210926110503_kurokiri_Combine.csv'
                  or csv_file_name == '20210926110605_kurokiri_Combine.csv'
                  or csv_file_name == '20210926110705_kurokiri_Combine.csv'
                  or csv_file_name == '20210926110811_kurokiri_Combine.csv'
                  or csv_file_name == '20210926110904_kurokiri_Combine.csv'
                  or csv_file_name == '20210926111005_kurokiri_Combine.csv'
                  or csv_file_name == '20210926111104_kurokiri_Combine.csv'
                  or csv_file_name == '20210926111204_kurokiri_Combine.csv'
                  or csv_file_name == '20210926111304_kurokiri_Combine.csv'
                  or csv_file_name == '20210926111404_kurokiri_Combine.csv'
                  or csv_file_name == '20210926111504_kurokiri_Combine.csv'
                  or csv_file_name == '20210926111604_kurokiri_Combine.csv'
                  or csv_file_name == '20210926111705_kurokiri_Combine.csv'
                  or csv_file_name == '20210926111804_kurokiri_Combine.csv'
                  or csv_file_name == '20210926111904_kurokiri_Combine.csv'
                  or csv_file_name == '20210926112003_kurokiri_Combine.csv'
                  or csv_file_name == '20210926112103_kurokiri_Combine.csv'
                  or csv_file_name == '20210926112204_kurokiri_Combine.csv'
                  or csv_file_name == '20210926112304_kurokiri_Combine.csv'
                  or csv_file_name == '20210926134904_kurokiri_Combine.csv'
                  or csv_file_name == '20210926135006_kurokiri_Combine.csv'
                  or csv_file_name == '20210926135103_kurokiri_Combine.csv'
                  or csv_file_name == '20210926135204_kurokiri_Combine.csv'
                  or csv_file_name == '20210926135303_kurokiri_Combine.csv'
                  or csv_file_name == '20210926135404_kurokiri_Combine.csv'
                  or csv_file_name == '20210926135504_kurokiri_Combine.csv'
                  or csv_file_name == '20210926135603_kurokiri_Combine.csv'
                  or csv_file_name == '20210926135703_kurokiri_Combine.csv'
                  or csv_file_name == '20210926135803_kurokiri_Combine.csv'
                  or csv_file_name == '20210926135910_kurokiri_Combine.csv'
                  or csv_file_name == '20210926140005_kurokiri_Combine.csv'
                  or csv_file_name == '20210926140104_kurokiri_Combine.csv'
                  or csv_file_name == '20210926140205_kurokiri_Combine.csv'
                  or csv_file_name == '20210926140304_kurokiri_Combine.csv'
                  or csv_file_name == '20210926140403_kurokiri_Combine.csv'
                  or csv_file_name == '20210926140504_kurokiri_Combine.csv'
                  or csv_file_name == '20210926140607_kurokiri_Combine.csv'
                  or csv_file_name == '20210926140703_kurokiri_Combine.csv'
                  or csv_file_name == '20210926140804_kurokiri_Combine.csv'
                  or csv_file_name == '20210930222303_kurokiri_Combine.csv'
                  or csv_file_name == '20210930222403_kurokiri_Combine.csv'
                  or csv_file_name == '20210930222503_kurokiri_Combine.csv'
                  or csv_file_name == '20210930222603_kurokiri_Combine.csv'
                  or csv_file_name == '20210930222703_kurokiri_Combine.csv'
                  or csv_file_name == '20210930222803_kurokiri_Combine.csv'
                  or csv_file_name == '20210930222904_kurokiri_Combine.csv'
                  or csv_file_name == '20210930223005_kurokiri_Combine.csv'
                  or csv_file_name == '20210930223103_kurokiri_Combine.csv'
                  or csv_file_name == '20210930223203_kurokiri_Combine.csv'
                  or csv_file_name == '20210930223303_kurokiri_Combine.csv'
                  or csv_file_name == '20210930223404_kurokiri_Combine.csv'
                  or csv_file_name == '20210930223503_kurokiri_Combine.csv'
                  or csv_file_name == '20210930223603_kurokiri_Combine.csv'
                  or csv_file_name == '20210930223703_kurokiri_Combine.csv'
                  or csv_file_name == '20211001002305_kurokiri_Combine.csv'
                  or csv_file_name == '20211001002408_kurokiri_Combine.csv'
                  or csv_file_name == '20211001002504_kurokiri_Combine.csv'
                  or csv_file_name == '20211001002606_kurokiri_Combine.csv'
                  or csv_file_name == '20211001002703_kurokiri_Combine.csv'
                  or csv_file_name == '20211001002804_kurokiri_Combine.csv'
                  or csv_file_name == '20211001002904_kurokiri_Combine.csv'
                  or csv_file_name == '20211001003005_kurokiri_Combine.csv'
                  or csv_file_name == '20211001003104_kurokiri_Combine.csv'
                  or csv_file_name == '20211001003202_kurokiri_Combine.csv'
                  or csv_file_name == '20211001003303_kurokiri_Combine.csv'
                  or csv_file_name == '20211001003403_kurokiri_Combine.csv'
                  or csv_file_name == '20211001003506_kurokiri_Combine.csv'
                  or csv_file_name == '20211001003602_kurokiri_Combine.csv'
                  or csv_file_name == '20211001003703_kurokiri_Combine.csv'
                  or csv_file_name == '20211002120405_kurokiri_Combine.csv'
                  or csv_file_name == '20211002120611_kurokiri_Combine.csv'
                  or csv_file_name == '20211002120805_kurokiri_Combine.csv'
                  or csv_file_name == '20211002121005_kurokiri_Combine.csv'
                  or csv_file_name == '20211002121205_kurokiri_Combine.csv'
                  or csv_file_name == '20211002121405_kurokiri_Combine.csv'
                  or csv_file_name == '20211002121605_kurokiri_Combine.csv'
                  or csv_file_name == '20211002121805_kurokiri_Combine.csv'
                  or csv_file_name == '20211002122005_kurokiri_Combine.csv'
                  or csv_file_name == '20211002122205_kurokiri_Combine.csv'
                  or csv_file_name == '20211002122405_kurokiri_Combine.csv'
                  or csv_file_name == '20211002122605_kurokiri_Combine.csv'
                  or csv_file_name == '20211002122805_kurokiri_Combine.csv'
                  or csv_file_name == '20211002123005_kurokiri_Combine.csv'
                  or csv_file_name == '20211002123205_kurokiri_Combine.csv'
                  or csv_file_name == '20211002150206_kurokiri_Combine.csv'
                  or csv_file_name == '20211002150405_kurokiri_Combine.csv'
                  or csv_file_name == '20211002150605_kurokiri_Combine.csv'
                  or csv_file_name == '20211002150805_kurokiri_Combine.csv'
                  or csv_file_name == '20211002151005_kurokiri_Combine.csv'
                  or csv_file_name == '20211002151205_kurokiri_Combine.csv'
                  or csv_file_name == '20211002151406_kurokiri_Combine.csv'
                  or csv_file_name == '20211002151605_kurokiri_Combine.csv'
                  or csv_file_name == '20211002151805_kurokiri_Combine.csv'
                  or csv_file_name == '20211002152006_kurokiri_Combine.csv'
                  or csv_file_name == '20211002152206_kurokiri_Combine.csv'
                  or csv_file_name == '20211002152405_kurokiri_Combine.csv'
                  or csv_file_name == '20211002152605_kurokiri_Combine.csv'
                  or csv_file_name == '20211002153005_kurokiri_Combine.csv'
                  or csv_file_name == '20211203195807_kurokiri_Combine.csv'  #shinchi
                  or csv_file_name == '20211204002238_kurokiri_Combine.csv'  #shinchi
                  or csv_file_name == '20211204002534_kurokiri_Combine.csv'  #shinchi
                  or csv_file_name == '20211204040436_kurokiri_Combine.csv'  #shinchi
                  or csv_file_name == '20211204102331_kurokiri_Combine.csv'  #shinchi
                  or csv_file_name == '20211205034234_kurokiri_Combine.csv'):  #shinchi
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 1, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_val is None:
                    x_val = x_timeseries
                    y_val = y_data
                else:
                    x_val = np.concatenate((x_val, x_timeseries), axis=0)
                    y_val = np.concatenate((y_val, y_data), axis=0)
                    
            elif (csv_file_name == '20210923144504_kurokiri_Combine.csv'
                  or csv_file_name == '20210923144606_kurokiri_Combine.csv'
                  or csv_file_name == '20210923144705_kurokiri_Combine.csv'
                  or csv_file_name == '20210923144804_kurokiri_Combine.csv'
                  or csv_file_name == '20210923144912_kurokiri_Combine.csv'
                  or csv_file_name == '20210923145005_kurokiri_Combine.csv'
                  or csv_file_name == '20210923145104_kurokiri_Combine.csv'
                  or csv_file_name == '20210923145204_kurokiri_Combine.csv'
                  or csv_file_name == '20210923145306_kurokiri_Combine.csv'
                  or csv_file_name == '20210923145403_kurokiri_Combine.csv'
                  or csv_file_name == '20210923145507_kurokiri_Combine.csv'
                  or csv_file_name == '20210923145604_kurokiri_Combine.csv'
                  or csv_file_name == '20210923145704_kurokiri_Combine.csv'
                  or csv_file_name == '20210923145805_kurokiri_Combine.csv'
                  or csv_file_name == '20210923145904_kurokiri_Combine.csv'
                  or csv_file_name == '20210923150004_kurokiri_Combine.csv'
                  or csv_file_name == '20210923150104_kurokiri_Combine.csv'
                  or csv_file_name == '20210923150205_kurokiri_Combine.csv'
                  or csv_file_name == '20210923150305_kurokiri_Combine.csv'
                  or csv_file_name == '20210923150404_kurokiri_Combine.csv'
                  or csv_file_name == '20210923150505_kurokiri_Combine.csv'
                  or csv_file_name == '20210923150604_kurokiri_Combine.csv'
                  or csv_file_name == '20210923150704_kurokiri_Combine.csv'
                  or csv_file_name == '20210923150804_kurokiri_Combine.csv'
                  or csv_file_name == '20210923150906_kurokiri_Combine.csv'
                  or csv_file_name == '20210923151004_kurokiri_Combine.csv'
                  or csv_file_name == '20210923151105_kurokiri_Combine.csv'
                  or csv_file_name == '20210923151205_kurokiri_Combine.csv'
                  or csv_file_name == '20210923151304_kurokiri_Combine.csv'
                  or csv_file_name == '20210923151405_kurokiri_Combine.csv'
                  or csv_file_name == '20210923151505_kurokiri_Combine.csv'
                  or csv_file_name == '20210923151604_kurokiri_Combine.csv'
                  or csv_file_name == '20210923151704_kurokiri_Combine.csv'
                  or csv_file_name == '20210923151806_kurokiri_Combine.csv'
                  or csv_file_name == '20210923151906_kurokiri_Combine.csv'
                  or csv_file_name == '20210923152007_kurokiri_Combine.csv'
                  or csv_file_name == '20210923152104_kurokiri_Combine.csv'
                  or csv_file_name == '20210923152205_kurokiri_Combine.csv'
                  or csv_file_name == '20210923152306_kurokiri_Combine.csv'
                  or csv_file_name == '20210923152407_kurokiri_Combine.csv'
                  or csv_file_name == '20210923194405_kurokiri_Combine.csv'
                  or csv_file_name == '20210923194505_kurokiri_Combine.csv'
                  or csv_file_name == '20210923194605_kurokiri_Combine.csv'
                  or csv_file_name == '20210923194704_kurokiri_Combine.csv'
                  or csv_file_name == '20210923194808_kurokiri_Combine.csv'
                  or csv_file_name == '20210923194904_kurokiri_Combine.csv'
                  or csv_file_name == '20210923195004_kurokiri_Combine.csv'
                  or csv_file_name == '20210923195105_kurokiri_Combine.csv'
                  or csv_file_name == '20210923195208_kurokiri_Combine.csv'
                  or csv_file_name == '20210923195305_kurokiri_Combine.csv'
                  or csv_file_name == '20210923195404_kurokiri_Combine.csv'
                  or csv_file_name == '20210923195504_kurokiri_Combine.csv'
                  or csv_file_name == '20210923195605_kurokiri_Combine.csv'
                  or csv_file_name == '20210923195706_kurokiri_Combine.csv'
                  or csv_file_name == '20210923195805_kurokiri_Combine.csv'
                  or csv_file_name == '20210923195905_kurokiri_Combine.csv'
                  or csv_file_name == '20210923200012_kurokiri_Combine.csv'
                  or csv_file_name == '20210923200105_kurokiri_Combine.csv'
                  or csv_file_name == '20210923200204_kurokiri_Combine.csv'
                  or csv_file_name == '20210923200304_kurokiri_Combine.csv'
                  or csv_file_name == '20210923200405_kurokiri_Combine.csv'
                  or csv_file_name == '20210923200504_kurokiri_Combine.csv'
                  or csv_file_name == '20210923200606_kurokiri_Combine.csv'
                  or csv_file_name == '20210923200706_kurokiri_Combine.csv'
                  or csv_file_name == '20210923200820_kurokiri_Combine.csv'
                  or csv_file_name == '20210923200904_kurokiri_Combine.csv'
                  or csv_file_name == '20210923201004_kurokiri_Combine.csv'
                  or csv_file_name == '20210923201107_kurokiri_Combine.csv'
                  or csv_file_name == '20210923201204_kurokiri_Combine.csv'
                  or csv_file_name == '20210923201304_kurokiri_Combine.csv'
                  or csv_file_name == '20210923201405_kurokiri_Combine.csv'
                  or csv_file_name == '20210923201504_kurokiri_Combine.csv'
                  or csv_file_name == '20210923201606_kurokiri_Combine.csv'
                  or csv_file_name == '20210923201704_kurokiri_Combine.csv'
                  or csv_file_name == '20210923201807_kurokiri_Combine.csv'
                  or csv_file_name == '20210923201904_kurokiri_Combine.csv'
                  or csv_file_name == '20210923202004_kurokiri_Combine.csv'
                  or csv_file_name == '20210923202104_kurokiri_Combine.csv'
                  or csv_file_name == '20210923202205_kurokiri_Combine.csv'
                  or csv_file_name == '20210923202305_kurokiri_Combine.csv'
                  or csv_file_name == '20210926112404_kurokiri_Combine.csv'
                  or csv_file_name == '20210926112506_kurokiri_Combine.csv'
                  or csv_file_name == '20210926112604_kurokiri_Combine.csv'
                  or csv_file_name == '20210926112704_kurokiri_Combine.csv'
                  or csv_file_name == '20210926112804_kurokiri_Combine.csv'
                  or csv_file_name == '20210926112903_kurokiri_Combine.csv'
                  or csv_file_name == '20210926113004_kurokiri_Combine.csv'
                  or csv_file_name == '20210926113104_kurokiri_Combine.csv'
                  or csv_file_name == '20210926113204_kurokiri_Combine.csv'
                  or csv_file_name == '20210926113304_kurokiri_Combine.csv'
                  or csv_file_name == '20210926113404_kurokiri_Combine.csv'
                  or csv_file_name == '20210926113504_kurokiri_Combine.csv'
                  or csv_file_name == '20210926113604_kurokiri_Combine.csv'
                  or csv_file_name == '20210926113704_kurokiri_Combine.csv'
                  or csv_file_name == '20210926113804_kurokiri_Combine.csv'
                  or csv_file_name == '20210926113904_kurokiri_Combine.csv'
                  or csv_file_name == '20210926114004_kurokiri_Combine.csv'
                  or csv_file_name == '20210926114104_kurokiri_Combine.csv'
                  or csv_file_name == '20210926114204_kurokiri_Combine.csv'
                  or csv_file_name == '20210926114304_kurokiri_Combine.csv'
                  or csv_file_name == '20210926132905_kurokiri_Combine.csv'
                  or csv_file_name == '20210926133004_kurokiri_Combine.csv'
                  or csv_file_name == '20210926133105_kurokiri_Combine.csv'
                  or csv_file_name == '20210926133207_kurokiri_Combine.csv'
                  or csv_file_name == '20210926133304_kurokiri_Combine.csv'
                  or csv_file_name == '20210926133404_kurokiri_Combine.csv'
                  or csv_file_name == '20210926133505_kurokiri_Combine.csv'
                  or csv_file_name == '20210926133610_kurokiri_Combine.csv'
                  or csv_file_name == '20210926133705_kurokiri_Combine.csv'
                  or csv_file_name == '20210926133804_kurokiri_Combine.csv'
                  or csv_file_name == '20210926133903_kurokiri_Combine.csv'
                  or csv_file_name == '20210926134005_kurokiri_Combine.csv'
                  or csv_file_name == '20210926134103_kurokiri_Combine.csv'
                  or csv_file_name == '20210926134204_kurokiri_Combine.csv'
                  or csv_file_name == '20210926134308_kurokiri_Combine.csv'
                  or csv_file_name == '20210926134405_kurokiri_Combine.csv'
                  or csv_file_name == '20210926134503_kurokiri_Combine.csv'
                  or csv_file_name == '20210926134609_kurokiri_Combine.csv'
                  or csv_file_name == '20210926134704_kurokiri_Combine.csv'
                  or csv_file_name == '20210926134803_kurokiri_Combine.csv'
                  or csv_file_name == '20210930220703_kurokiri_Combine.csv'
                  or csv_file_name == '20210930220804_kurokiri_Combine.csv'
                  or csv_file_name == '20210930220921_kurokiri_Combine.csv'
                  or csv_file_name == '20210930221114_kurokiri_Combine.csv'
                  or csv_file_name == '20210930221203_kurokiri_Combine.csv'
                  or csv_file_name == '20210930221305_kurokiri_Combine.csv'
                  or csv_file_name == '20210930221411_kurokiri_Combine.csv'
                  or csv_file_name == '20210930221504_kurokiri_Combine.csv'
                  or csv_file_name == '20210930221603_kurokiri_Combine.csv'
                  or csv_file_name == '20210930221703_kurokiri_Combine.csv'
                  or csv_file_name == '20210930221803_kurokiri_Combine.csv'
                  or csv_file_name == '20210930221904_kurokiri_Combine.csv'
                  or csv_file_name == '20210930222003_kurokiri_Combine.csv'
                  or csv_file_name == '20210930222106_kurokiri_Combine.csv'
                  or csv_file_name == '20210930222203_kurokiri_Combine.csv'
                  or csv_file_name == '20211001003804_kurokiri_Combine.csv'
                  or csv_file_name == '20211001003904_kurokiri_Combine.csv'
                  or csv_file_name == '20211001004004_kurokiri_Combine.csv'
                  or csv_file_name == '20211001004102_kurokiri_Combine.csv'
                  or csv_file_name == '20211001004205_kurokiri_Combine.csv'
                  or csv_file_name == '20211001004303_kurokiri_Combine.csv'
                  or csv_file_name == '20211001004403_kurokiri_Combine.csv'
                  or csv_file_name == '20211001004503_kurokiri_Combine.csv'
                  or csv_file_name == '20211001004603_kurokiri_Combine.csv'
                  or csv_file_name == '20211001004703_kurokiri_Combine.csv'
                  or csv_file_name == '20211001004803_kurokiri_Combine.csv'
                  or csv_file_name == '20211001004903_kurokiri_Combine.csv'
                  or csv_file_name == '20211001005009_kurokiri_Combine.csv'
                  or csv_file_name == '20211001005103_kurokiri_Combine.csv'
                  or csv_file_name == '20211001005205_kurokiri_Combine.csv'
                  or csv_file_name == '20211002113405_kurokiri_Combine.csv'
                  or csv_file_name == '20211002113607_kurokiri_Combine.csv'
                  or csv_file_name == '20211002113805_kurokiri_Combine.csv'
                  or csv_file_name == '20211002114005_kurokiri_Combine.csv'
                  or csv_file_name == '20211002114204_kurokiri_Combine.csv'
                  or csv_file_name == '20211002114411_kurokiri_Combine.csv'
                  or csv_file_name == '20211002114605_kurokiri_Combine.csv'
                  or csv_file_name == '20211002114805_kurokiri_Combine.csv'
                  or csv_file_name == '20211002115004_kurokiri_Combine.csv'
                  or csv_file_name == '20211002115204_kurokiri_Combine.csv'
                  or csv_file_name == '20211002115411_kurokiri_Combine.csv'
                  or csv_file_name == '20211002115605_kurokiri_Combine.csv'
                  or csv_file_name == '20211002115805_kurokiri_Combine.csv'
                  or csv_file_name == '20211002120005_kurokiri_Combine.csv'
                  or csv_file_name == '20211002120205_kurokiri_Combine.csv'
                  or csv_file_name == '20211002153205_kurokiri_Combine.csv'
                  or csv_file_name == '20211002153405_kurokiri_Combine.csv'
                  or csv_file_name == '20211002153606_kurokiri_Combine.csv'
                  or csv_file_name == '20211002153805_kurokiri_Combine.csv'
                  or csv_file_name == '20211002154005_kurokiri_Combine.csv'
                  or csv_file_name == '20211002154205_kurokiri_Combine.csv'
                  or csv_file_name == '20211002154405_kurokiri_Combine.csv'
                  or csv_file_name == '20211002154605_kurokiri_Combine.csv'
                  or csv_file_name == '20211002154805_kurokiri_Combine.csv'
                  or csv_file_name == '20211002155005_kurokiri_Combine.csv'
                  or csv_file_name == '20211002155206_kurokiri_Combine.csv'
                  or csv_file_name == '20211002155405_kurokiri_Combine.csv'
                  or csv_file_name == '20211002155605_kurokiri_Combine.csv'
                  or csv_file_name == '20211002155806_kurokiri_Combine.csv'
                  or csv_file_name == '20211002160006_kurokiri_Combine.csv'
                  or csv_file_name == '20211203200108_kurokiri_Combine.csv'  #shinchi
                  or csv_file_name == '20211204004337_kurokiri_Combine.csv'  #shinchi
                  or csv_file_name == '20211204005531_kurokiri_Combine.csv'  #shinchi
                  or csv_file_name == '20211204035832_kurokiri_Combine.csv'  #shinchi
                  or csv_file_name == '20211204102629_kurokiri_Combine.csv'  #shinchi
                  or csv_file_name == '20211205040025_kurokiri_Combine.csv'):  #shinchi
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 1, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_test is None:
                    x_test = x_timeseries
                    y_test = y_data
                else:
                    x_test = np.concatenate((x_test, x_timeseries), axis=0)
                    y_test = np.concatenate((y_test, y_data), axis=0)
                    
            elif 'kurokiri_Combine.csv' in csv_file_name:
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 1, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_train is None:
                    x_train = x_timeseries
                    y_train = y_data
                else:
                    x_train = np.concatenate((x_train, x_timeseries), axis=0)
                    y_train = np.concatenate((y_train, y_data), axis=0)
            else:
                pass
            
# -----------------------------------------------------------------------------
    
        elif pattern_2:
            if (csv_file_name == '20210824172605_coffee_Combine.csv' 
                  or csv_file_name == '20210824172703_coffee_Combine.csv'
                  or csv_file_name == '20210826113724_coffee_Combine.csv'
                  or csv_file_name == '20210826113825_coffee_Combine.csv'
                  or csv_file_name == '20210826103704_coffee_Combine.csv'
                  or csv_file_name == '20210826103804_coffee_Combine.csv'):
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 0, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_val is None:
                    x_val = x_timeseries
                    y_val = y_data
                else:
                    x_val = np.concatenate((x_val, x_timeseries), axis=0)
                    y_val = np.concatenate((y_val, y_data), axis=0)
                    
            elif (csv_file_name == '20210824172804_coffee_Combine.csv' 
                  or csv_file_name == '20210824172904_coffee_Combine.csv'
                  or csv_file_name == '20210826113924_coffee_Combine.csv'
                  or csv_file_name == '20210826114024_coffee_Combine.csv'
                  or csv_file_name == '20210826103905_coffee_Combine.csv'
                  or csv_file_name == '20210826104004_coffee_Combine.csv'):
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 0, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_test is None:
                    x_test = x_timeseries
                    y_test = y_data
                else:
                    x_test = np.concatenate((x_test, x_timeseries), axis=0)
                    y_test = np.concatenate((y_test, y_data), axis=0)
                    
            elif 'coffee_Combine.csv' in csv_file_name:
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 0, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_train is None:
                    x_train = x_timeseries
                    y_train = y_data
                else:
                    x_train = np.concatenate((x_train, x_timeseries), axis=0)
                    y_train = np.concatenate((y_train, y_data), axis=0)
                    
            elif (csv_file_name == '20210823150804_kurokiri_Combine.csv' 
                  or csv_file_name == '20210823150901_kurokiri_Combine.csv'
                  or csv_file_name == '20210824165803_kurokiri_Combine.csv'
                  or csv_file_name == '20210824165904_kurokiri_Combine.csv'
                  or csv_file_name == '20210824142003_kurokiri_Combine.csv'
                  or csv_file_name == '20210824142103_kurokiri_Combine.csv'):
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 1, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_val is None:
                    x_val = x_timeseries
                    y_val = y_data
                else:
                    x_val = np.concatenate((x_val, x_timeseries), axis=0)
                    y_val = np.concatenate((y_val, y_data), axis=0)
                    
            elif (csv_file_name == '20210823151002_kurokiri_Combine.csv' 
                  or csv_file_name == '20210823151102_kurokiri_Combine.csv'
                  or csv_file_name == '20210824170003_kurokiri_Combine.csv'
                  or csv_file_name == '20210824170104_kurokiri_Combine.csv'
                  or csv_file_name == '20210824142202_kurokiri_Combine.csv'
                  or csv_file_name == '20210824142304_kurokiri_Combine.csv'):
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 1, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_test is None:
                    x_test = x_timeseries
                    y_test = y_data
                else:
                    x_test = np.concatenate((x_test, x_timeseries), axis=0)
                    y_test = np.concatenate((y_test, y_data), axis=0)
                    
            elif 'kurokiri_Combine.csv' in csv_file_name:
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 1, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_train is None:
                    x_train = x_timeseries
                    y_train = y_data
                else:
                    x_train = np.concatenate((x_train, x_timeseries), axis=0)
                    y_train = np.concatenate((y_train, y_data), axis=0)
            else:
                pass
            
# -----------------------------------------------------------------------------

        elif pattern_3:
            if (csv_file_name == '20210824172804_coffee_Combine.csv' 
                  or csv_file_name == '20210824172904_coffee_Combine.csv'
                  or csv_file_name == '20210826113924_coffee_Combine.csv'
                  or csv_file_name == '20210826114024_coffee_Combine.csv'
                  or csv_file_name == '20210826103905_coffee_Combine.csv'
                  or csv_file_name == '20210826104004_coffee_Combine.csv'):
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 0, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_val is None:
                    x_val = x_timeseries
                    y_val = y_data
                else:
                    x_val = np.concatenate((x_val, x_timeseries), axis=0)
                    y_val = np.concatenate((y_val, y_data), axis=0)
                    
            elif (csv_file_name == '20210824173003_coffee_Combine.csv' 
                  or csv_file_name == '20210824173104_coffee_Combine.csv'
                  or csv_file_name == '20210826114123_coffee_Combine.csv'
                  or csv_file_name == '20210826114224_coffee_Combine.csv'
                  or csv_file_name == '20210826104104_coffee_Combine.csv'
                  or csv_file_name == '20210826104204_coffee_Combine.csv'):
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 0, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_test is None:
                    x_test = x_timeseries
                    y_test = y_data
                else:
                    x_test = np.concatenate((x_test, x_timeseries), axis=0)
                    y_test = np.concatenate((y_test, y_data), axis=0)
                    
            elif 'coffee_Combine.csv' in csv_file_name:
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 0, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_train is None:
                    x_train = x_timeseries
                    y_train = y_data
                else:
                    x_train = np.concatenate((x_train, x_timeseries), axis=0)
                    y_train = np.concatenate((y_train, y_data), axis=0)
                    
            elif (csv_file_name == '20210823151002_kurokiri_Combine.csv' 
                  or csv_file_name == '20210823151102_kurokiri_Combine.csv'
                  or csv_file_name == '20210824170003_kurokiri_Combine.csv'
                  or csv_file_name == '20210824170104_kurokiri_Combine.csv'
                  or csv_file_name == '20210824142202_kurokiri_Combine.csv'
                  or csv_file_name == '20210824142304_kurokiri_Combine.csv'):
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 1, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_val is None:
                    x_val = x_timeseries
                    y_val = y_data
                else:
                    x_val = np.concatenate((x_val, x_timeseries), axis=0)
                    y_val = np.concatenate((y_val, y_data), axis=0)
                    
            elif (csv_file_name == '20210823151202_kurokiri_Combine.csv' 
                  or csv_file_name == '20210823151302_kurokiri_Combine.csv'
                  or csv_file_name == '20210824170204_kurokiri_Combine.csv'
                  or csv_file_name == '20210824170304_kurokiri_Combine.csv'
                  or csv_file_name == '20210824142403_kurokiri_Combine.csv'
                  or csv_file_name == '20210824142503_kurokiri_Combine.csv'):
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 1, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_test is None:
                    x_test = x_timeseries
                    y_test = y_data
                else:
                    x_test = np.concatenate((x_test, x_timeseries), axis=0)
                    y_test = np.concatenate((y_test, y_data), axis=0)
                    
            elif 'kurokiri_Combine.csv' in csv_file_name:
                x_timeseries = np.ones( (reader.shape[0]-timeseries_num+1, timeseries_num, reader.shape[-1]), dtype='int32')
                for j in range(reader.shape[0]-timeseries_num+1):
                    x_timeseries_temp = reader[j:j+timeseries_num, :].reshape(1,timeseries_num,reader.shape[-1])
                    x_timeseries[j] = x_timeseries_temp
                pre_y_data = np.full( (reader.shape[0]-timeseries_num+1, 1), 1, dtype='int16')
                y_data = to_categorical(pre_y_data, num_classes=class_num, dtype='int16')
                if x_train is None:
                    x_train = x_timeseries
                    y_train = y_data
                else:
                    x_train = np.concatenate((x_train, x_timeseries), axis=0)
                    y_train = np.concatenate((y_train, y_data), axis=0)
            else:
                pass
            
# ----------------------------------------------------------------------------- 

if scaling_1div100:
    x_train = x_train/100
    x_val = x_val/100
    x_test = x_test/100
    
print('\nx_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_val.shape = ', x_val.shape)
print('y_val.shape = ', y_val.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)  
    
# -----------------------------------------------------------------------------

def tcn_baseline_1(input_shape=(x_train.shape[1], 10), classes=class_num):
    i = Input(shape=input_shape, name='input_elements')
    
    o = tcn.TCN(nb_filters=32, dilations=(1, 2, 4, 8), 
                use_batch_norm=True, return_sequences=False)(i)
    o = Dense(classes, name='dense_output')(o)
    cla_output = Activation('sigmoid')(o)
    model = Model(inputs=i, outputs=cla_output)
    
    return model

# -----------------------------------------------------------------------------

def tcn_baseline_2(input_shape=(x_train.shape[1], 10), classes=class_num):
    i = Input(shape=input_shape, name='input_elements')
    
    o = tcn.TCN(nb_filters=8, kernel_size=2, dilations=(1, 2, 4), 
                use_batch_norm=True, return_sequences=False)(i)
    o = Dense(classes, name='dense_output')(o)
    cla_output = Activation('sigmoid')(o)
    model = Model(inputs=i, outputs=cla_output)
    
    return model

# -----------------------------------------------------------------------------

def tcn_baseline_3(input_shape=(x_train.shape[1], 10), classes=class_num):
    i = Input(shape=input_shape, name='input_elements')
    
    o = tcn.TCN(nb_filters=8, kernel_size=2, dilations=(1, 2, 4), 
                use_batch_norm=True, return_sequences=True)(i)
    o = tcn.TCN(nb_filters=8, kernel_size=2, dilations=(1, 2, 4), 
                use_batch_norm=True, return_sequences=False)(o)
    o = Dense(classes, name='dense_output')(o)
    cla_output = Activation('sigmoid')(o)
    model = Model(inputs=i, outputs=cla_output)
    
    return model
 
# -----------------------------------------------------------------------------

def lstm_baseline(input_shape=(x_train.shape[1], 10), classes=class_num, filter1=4):
    inputs = Input(shape=input_shape, name='input_elements')
    
    x = keras.layers.TimeDistributed(Dense(filter1*2))(inputs)
    x = keras.layers.TimeDistributed(Activation('relu'))(x)
    x = keras.layers.TimeDistributed(BatchNormalization())(x)
    
    x = keras.layers.LSTM(filter1*4, return_sequences=False)(x)
    #x = keras.layers.RNN(tfa.rnn.LayerNormLSTMCell(filter1*4), return_sequences=True)(x)
    #x = keras.layers.RNN(tfa.rnn.LayerNormLSTMCell(filter1*1), return_sequences=False)(x)
    
    x = Dense(classes, name='dense_output')(x)
    cla_output = Activation('sigmoid')(x)
    model = Model(inputs=inputs, outputs=cla_output)
    
    return model

# -----------------------------------------------------------------------------

def hist_plot(hist):
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    fig, ax1 = plt.subplots(figsize=(10,10))
    ax1.plot(hist.history['accuracy'], color='#2ca02c', label='train acc.')
    ax1.plot(hist.history['val_accuracy'], color='#d62728', label='val. acc.')
    ax1.set_ylim([0.0,1.1])
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig('./result_graph.png')
    
# -----------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    cmx = confusion_matrix(y_true, y_pred)
    print(cmx)
    if normalize:
        cmx = cmx.astype('float') / cmx.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cmx)
    fig, ax = plt.subplots()
    im = ax.imshow(cmx, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cmx.shape[1]),
           yticks=np.arange(cmx.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cmx.max() / 2.
    for i in range(cmx.shape[0]):
        for j in range(cmx.shape[1]):
            ax.text(j, i, format(cmx[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cmx[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('./result_confusion_matrix.png')
    return ax

# -----------------------------------------------------------------------------

if learning_mode:
    if tcn_baseline_1:
        model = tcn_baseline_1(input_shape=(x_train.shape[1], 10), classes=class_num)
    elif tcn_baseline_2:
        model = tcn_baseline_2(input_shape=(x_train.shape[1], 10), classes=class_num)
    elif tcn_baseline_3:
        model = tcn_baseline_3(input_shape=(x_train.shape[1], 10), classes=class_num)
    elif lstm_baseline:
        model = lstm_baseline(input_shape=(x_train.shape[1], 10), classes=class_num)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(), metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=patience, verbose=1, mode='max',
                                   baseline=None, restore_best_weights=True)
    if earlystopping:
        hist = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                         batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[early_stopping])
    else:
        hist = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                         batch_size=batch_size, epochs=epochs, verbose=1)
    with open('./hist.pkl', mode='wb') as fw:
        pickle.dump(hist.history, fw)
        print('\nsaved hist.')
    hist_plot(hist)
    model.save('C:/Users/liamc/Desktop/model_20220210/model_20220203/model+weights_best.h5')
    print('\nsaved model+weights_best.')
    
# -----------------------------------------------------------------------------

elif evaluation_mode:
    model = keras.models.load_model('./model+weights_best.h5', custom_objects={'TCN': tcn.TCN})
    score_dist = model.evaluate(x_test, y_test, verbose=1)
    #hist = pickle.load(open('./hist.pkl','rb'))

    y_pred = model.predict(x_test)
    y_test_dec = np.full((y_test.shape[0], ), 0, dtype='int16')
    for i in range(y_test.shape[0]):
        y_test_dec[i] = np.argmax(y_test[i])
    y_pred_dec = np.full((y_pred.shape[0], ), 0, dtype='int16')
    for i in range(y_pred.shape[0]):
        y_pred_dec[i] = np.argmax(y_pred[i])
    y_test_dec = y_test_dec.tolist()
    y_pred_dec = y_pred_dec.tolist()
    class_names=['coffee', 'alcohol']
    #print('\ny_test_dec = ', y_test_dec)
    #print('y_pred_dec = ', y_pred_dec)
    #print('class_names = ', class_names)
    plot_confusion_matrix(y_true=y_test_dec, y_pred=y_pred_dec, classes=class_names, normalize=True)

    model = Model(inputs=model.get_layer('input_elements').input, outputs=model.get_layer('dense_output').output)
    gallery_model = model.predict(x_train, verbose=1)
    gallery_model /= np.linalg.norm(gallery_model, ord=2, axis=1)[:, np.newaxis]
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(gallery_model, y_train)
    embedding = TSNE(n_components=2).fit_transform(gallery_model)
    
    y_train_enc = np.full((y_train.shape[0], ), 0, dtype='int16')
    for i in range(y_train.shape[0]):
        for j in range(y_train.shape[1]):
            if y_train[i][j] == 0:
                continue
            elif y_train[i][j] == 1:
                y_train_enc[i] = j+1
    fig = plt.figure(figsize=(30,20))
    ax = fig.add_subplot(1,1,1)
    plt.scatter(embedding[:,0], embedding[:,1], c=y_train_enc, cmap=cm.tab20)
    ax.set_aspect('equal')
    plt.colorbar()
    fig.savefig('./result_t-sne.png')
    
    probe_model = model.predict(x_test, verbose=1)
    probe_model /= np.linalg.norm(probe_model, ord=2, axis=1)[:, np.newaxis]
    pred_model = knn.predict(probe_model)
    score_knn = accuracy_score(y_test, pred_model)
    print('\nscore_dist loss : ', score_dist[0])
    print('score_dist acc. : ', score_dist[1])
    print('score_knn = ', score_knn)
    
# -----------------------------------------------------------------------------

elif prediction_mode:
    model = keras.models.load_model('./model+weights_best.h5', custom_objects={'TCN': tcn.TCN})  # ground truth : coke -> 0, coffee -> 1, alcochol -> 2
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred,axis=1)
    y_pred_name = []
    for i in range(y_pred_classes.shape[0]):
        if y_pred_classes[i]==0:
           y_pred_name.append(str('coffee'))
        elif y_pred_classes[i]==1:
           y_pred_name.append(str('alcohol'))
    y_test_classes = np.argmax(y_test, axis=1)
    y_test_name = []
    for i in range(y_test_classes.shape[0]):
        if y_test_classes[i]==0:
           y_test_name.append(str('coffee'))
        elif y_test_classes[i]==1:
           y_test_name.append(str('alcohol'))
    for i in range(len(y_pred_name)):
        print('True : ' + str(y_test_name[i]) + ', Predict : ' + str(y_pred_name[i]))
        
# -----------------------------------------------------------------------------

















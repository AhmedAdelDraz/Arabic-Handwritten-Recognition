import os 
# os.environ['TF_CPP_MIN_LOG_LEVEL']

import cv2
import numpy as np
import string
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow.keras.backend as K

from tensorflow import keras 
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,Reshape,BatchNormalization,Input,Conv2D,MaxPool2D,Lambda,Bidirectional
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical,Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from collections import Counter
from PIL import Image
from itertools import groupby
from pathlib import Path
from keras import losses
import utils

class CTCLoss(losses.Loss):

  def __init__(self):
      super().__init__()
      self.loss_fn = keras.backend.ctc_batch_cost

  def call(self, y_true, y_pred):
      batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
      input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
      label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

      input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
      label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

      loss = self.loss_fn(y_true, y_pred, input_length, label_length)
      return loss
        
class CTCDecoder:
  def __init__(self):
    # self.predictions = predictions
    self.text_list = []
    # self.decode()

  def decode(self,predictions,char_list):
    pred_indcies = np.argmax(predictions, axis=2)
    for i in range(pred_indcies.shape[0]):
        ans = ""
        ## merge repeats
        merged_list = [k for k,_ in groupby(pred_indcies[i])]
        ## remove blanks
        for p in merged_list:
            if p != len(char_list):
                ans += char_list[int(p)]
        self.text_list.append(ans)
    return self.text_list


class CNN(Model):
  def __init__(self):
    super(CNN,self).__init__()
    self.conv_1 = Conv2D(32, (3,3), activation = "selu", padding='same')
    self.pool_1 = MaxPool2D(pool_size=(2, 2))
    
    self.conv_2 = Conv2D(64, (3,3), activation = "selu", padding='same')
    self.pool_2 = MaxPool2D(pool_size=(2, 2))

    self.conv_3 = Conv2D(128, (3,3), activation = "selu", padding='same')
    self.conv_4 = Conv2D(128, (3,3), activation = "selu", padding='same')

    self.pool_4 = MaxPool2D(pool_size=(2, 1))
    
    self.conv_5 = Conv2D(256, (3,3), activation = "selu", padding='same')
    
    # Batch normalization layer
    self.batch_norm_5 = BatchNormalization()
    
    self.conv_6 = Conv2D(256, (3,3), activation = "selu", padding='same')
    self.batch_norm_6 = BatchNormalization()
    self.pool_6 = MaxPool2D(pool_size=(2, 1))
    
    self.conv_7 = Conv2D(64, (2,2), activation = "selu")
  def call(self,inputs):
    x = self.conv_1(inputs)
    x = self.pool_1(x)
    x = self.conv_2(x)
    x = self.pool_2(x)
    x = self.conv_3(x)
    x = self.conv_4(x)
    x = self.pool_4(x)
    x = self.conv_5(x)
    x = self.batch_norm_5(x)
    x = self.conv_6(x)
    x = self.batch_norm_6(x)
    x = self.pool_6(x)
    x = self.conv_7(x)
    return x    


class RNN(Model):
  def __init__(self,depth,units,n_class):
    super(RNN,self).__init__()
    self.hidden = [Bidirectional(CuDNNLSTM(units, return_sequences=True)) for _ in range(depth)]
    self.softmax_output = Dense(n_class, activation = 'softmax')

  def call(self,inputs):
    x = inputs
    for layer in self.hidden:
      x = layer(x)
    x = self.softmax_output(x)
    return x

class HYBRID(Model):
  def __init__(self,n_class):
    super(HYBRID,self).__init__()
    self.squeezed = Lambda(lambda x: K.squeeze(x, 1))
    self.cnn = CNN()
    self.rnn = RNN(2,128,n_class)

  def call(self,inputs):
    x = self.cnn(inputs)
    x = self.squeezed(x)
    x = self.rnn(x)
    return x

# if __name__=='__main__':
#     train_dataset, validation_dataset, char_list = utils.preprocessing('./mjsynth_sample')
#     model = HYBRID()
#     optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
#     model.compile(loss=CTCLoss(),optimizer = optimizer)
#     # model.summary()
#     # print(next(iter(train_dataset)))
#     callbacks_list = [EarlyStopping(patience=3, verbose=1)]

#     history = model.fit(train_dataset, 
#                         epochs = 30,
#                         validation_data=validation_dataset,
#                         verbose = 1,
#                         callbacks = callbacks_list,
#                         shuffle=True)



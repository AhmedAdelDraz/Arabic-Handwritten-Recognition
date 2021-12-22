import os 
# os.environ['TF_CPP_MIN_LOG_LEVEL']

import cv2
import numpy as np
import string
# import pandas as pd 
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

##############
from utils import *
from model import *
import argparse


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--training_data','-t', type=str,required=True)
  parser.add_argument('--train_size','-s', type=float)
  args = parser.parse_args()
  data_folder = args.training_data
  train_size = args.train_size

  train_dataset, validation_dataset, char_list = preprocessing(data_folder,train_size)
  model = HYBRID(n_class=len(char_list)+1)
  optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
  model.compile(loss=CTCLoss(),optimizer = optimizer)

  checkpoint_path = "checkpoints/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                  monitor='val_accuracy',
                                                  save_weights_only=True,
                                                  save_best_only=True,
                                                  verbose=1)

  callbacks_list = [cp_callback,EarlyStopping(patience=3, verbose=1)]

  history = model.fit(train_dataset, 
                      epochs = 30,
                      validation_data=validation_dataset,
                      verbose = 1,
                      callbacks = callbacks_list,
                      shuffle=True)


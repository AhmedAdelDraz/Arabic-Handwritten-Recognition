import os
import cv2
import string
import tensorflow as tf 
from pathlib import Path
from functools import partial
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse

def load_data(directory_path):
    image_paths = []
    image_texts = []
    for path in os.listdir(directory_path):
        image_paths.append(directory_path + "/" + path)
        image_texts.append(path.split("_")[1])
    return image_paths, image_texts

# print(image_paths[:10], image_texts[:10])


def corrupt_check(image_paths,image_texts):
    corrupt_images = []
    for path in image_paths:
        try:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        except:
            corrupt_images.append(path)
    
    for path in corrupt_images:
        corrupt_index = image_paths.index(path)
        del image_paths[corrupt_index]
        del image_texts[corrupt_index]


def encode_to_labels(txt,char_list,max_label_len):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
    return pad_sequences([dig_lst], maxlen=max_label_len, padding='post', value=len(char_list))[0]



def process_single_sample(img_path, label):
    img = tf.io.read_file(img_path)                                 # 1. Read image
    img = tf.io.decode_png(img, channels=1)                         # 2. Decode and convert to grayscale
    img = tf.image.convert_image_dtype(img, tf.float32)             # 3. Convert to float32 in [0, 1] range
    img = tf.image.resize(img, [32, 128])                           # 4. Resize to the desired size
    return img,label


def preprocessing(data_folder,train_size):
    image_paths, image_texts = load_data(data_folder)
    corrupt_check(image_paths,image_texts)
    
    #labels preprocessing
    vocab = set("".join(map(str, image_texts)))                     # 1. Make a set contains all charts of the corpus
    char_list = sorted(vocab)                                 
    max_label_len = max([len(str(text)) for text in image_texts])
    encode_to_labels_mod = partial(encode_to_labels,char_list=char_list,max_label_len=max_label_len)   
    padded_image_texts = list(map(encode_to_labels_mod, image_texts))   # 2. Pad the each text label up to max length caculated above
    #image preprocessing
    train_image_paths = image_paths[ : int(len(image_paths) * train_size)]
    train_image_texts = padded_image_texts[ : int(len(image_texts) * train_size)]

    val_image_paths = image_paths[int(len(image_paths) * train_size) : ]
    val_image_texts = padded_image_texts[int(len(image_texts) * train_size) : ]
    
    batch_size = 256

    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_texts))

    train_dataset = (
        train_dataset.map(
            process_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_image_texts))
    validation_dataset = (
        validation_dataset.map(
            process_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    return train_dataset, validation_dataset, char_list
    

# if __name__=='__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument('-train', type=str,required=True)
#   args = parser.parse_args()
#   data_folder = args.train

#   train_dataset, validation_dataset, char_list = preprocessing(data_folder)
#   print(next(iter(train_dataset)))
# -*- coding: utf-8 -*-
"""ExactBuyer_Seniority-Classifier.ipynb

Made by Andrew Wood

Original file is located at
    https://colab.research.google.com/drive/1hB6do-ZRF_hW3hGjTKxdaPNfDpEzsUHO
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_recall_fscore_support as score

# This function will take an array of string labels in the format provided and return a NumPy array
#    of the seniority level for that label.
def vectorize_labels(labels):
    seniority_dict = {"'unpaid'" : 0, "'entry'" : 1, "'training'" : 2, "'senior'" : 3, "'manager'" : 4, 
                            "'director'" : 5, "'partner'" : 6, "'vp'" : 7, "'cxo'" : 8, "'owner'" : 9}
    y = []
    for i in range(len(labels)):
        s_level = labels[i]
        s_level = tf.keras.preprocessing.text.text_to_word_sequence(s_level)
        for idx, title in enumerate(s_level):
            title = seniority_dict[title]
            s_level[idx] = title
        y.append(max(s_level))

    y = np.array(y)
    return y

# This function will take an array of strings and return a tokenized, padded version of the text
def data_to_padded(X):
    sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(sequences, padding = 'post', maxlen= max_length)
    X_padded = np.array(X_padded)
    return X_padded

# This will build a deep learning model based on the parameters provided
def build_model(embedding_dim, vocab_size, max_length):
    model = tf.keras.Sequential([ 
                             tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
                             tf.keras.layers.GlobalAveragePooling1D(),
                             tf.keras.layers.Dense(640, activation='relu'),
                             tf.keras.layers.Dense(80, activation='relu'),
                             tf.keras.layers.Dense(10, activation='softmax')
    ])  
    return model

# Evaluate will take a csv file path as input, process it, use the trained model to predict labels on
#   the given data, then prints some performance metrics. 
def evaluate(csv_path):
    df = pd.read_csv(csv_path, header=0)
    seniority_levels = df['current_seniority_levels']
    inputs = df['current_company_industry'] + ' ' + df['current_title']

    new_padded = data_to_padded(inputs)
    y_actual = vectorize_labels(seniority_levels)

    #calculate predictions and metrics
    raw_predictions = model.predict(new_padded)
    yhat = []
    for prediction in raw_predictions:
        yhat.append(np.argmax(prediction))
    yhat = np.array(yhat)

    precision, recall, fscore, support = score(yhat, y_actual)
    print("\nThe following metrics are by calculated class and in ascending seniority order.\n")
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1-score: {}'.format(fscore))

    return

#CSV FILE PATHS TO CHANGE
file_path = "/content/drive/MyDrive/Datasets/ExactBuyer/train.csv"
csv_path = file_path

df = pd.read_csv(file_path, header=0)
seniority_levels = df['current_seniority_levels']
inputs = df['current_company_industry'] + ' ' + df['current_title']

embedding_dim = 1280
vocab_size = 1927
max_length = 17
num_epochs = 4
training_size = 6400 #1:4 split test:train

labels = vectorize_labels(seniority_levels)

x_train = inputs[0:training_size]
x_test = inputs[training_size:]

y_train = labels[0:training_size]
y_test = labels[training_size:]

tokenizer = Tokenizer(num_words = vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)

training_padded = data_to_padded(x_train)
testing_padded = data_to_padded(x_test)

model = build_model(embedding_dim, vocab_size, max_length)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(training_padded, y_train, epochs = num_epochs, validation_data=(testing_padded, y_test), verbose = 1)

evaluate(csv_path)
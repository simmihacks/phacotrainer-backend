"""
This script takes a single .h5 file representing the 
video input and runs the phacotrainer model locally
to predict the labels per 90 second segment.

Model Output Details
------------------------------------------------

14 possible labels = [
    'create wounds',
    'anterior chamber injection',
    'capsulorhexis',
    'hydrodissection',
    'phacoemulsification',
    'irrigation/aspiration',
    'place lens',
    'remove viscoelastic',
    'close wounds',
    'advanced technique',
    'subconjunctival injection',
    'tryptan blue staining',
    'iris manipulation',
    'no label'
]

Model Expected Input:
num_frames x (256 x 456 x 3)

Preprocessing for .h5 input:
green_channel_mean = sum(img[:, :, 0]) / (256 * 256)
red_channel_mean = sum(img[:, :, 1]) / (256 * 256)
blue_channel_mean = sum(img[:, :, 2]) / (256 * 256)

X = img[:, 0:256, 100:356, :] - [green_channel_mean, red_channel_mean, blue_channel_mean]

Numpy model output:
[
    # model output 1
    [0, 0, ..., 1, 0, 0, ...],
    [0, 0, ..., 1, 0, 0, ...],
    .... x90

    # model output 2
    [0, 0, ..., 1, 0, 0, ...],
    [0, 0, ..., 1, 0, 0, ...],
    .... x90
]

What we would save in db:
{
    "videoId": 1234,
    "results": [
        {start_time: i * segment_time, end_time: i+1 * segment_time, label: ""},
        {start_time: i * segment_time, end_time: i+1 * segment_time, label: ""},
        ...
        {start_time: i * segment_time, end_time: i+1 * segment_time, label: ""},
    ]
}

@author: Simmi Sen
@date: 05/31/2022
"""

import os
import tensorflow as tf
import keras
import numpy as np
import h5py
import random
import cv2
import json
from sklearn.metrics import classification_report
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, LSTM, Flatten, TimeDistributed, Conv2D, Dropout, AveragePooling2D, concatenate, Reshape, Lambda, Bidirectional
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.utils import Sequence

tf.keras.backend.set_image_data_format('channels_last')


class PhacotrainerModel(object):

    def __init__(self):
        self.h5_path = "h5_files"
        self.model_path = "model_weights"
        self.model_name = "best_model.h5"

        self.fps = 90
        self.height = 256
        self.width = 456
        self.num_labels = 14
        self.label_names = [
            'create wounds',
            'anterior chamber injection',
            'capsulorhexis',
            'hydrodissection',
            'phacoemulsification',
            'irrigation/aspiration',
            'place lens',
            'remove viscoelastic',
            'close wounds',
            'advanced technique',
            'subconjunctival injection',
            'tryptan blue staining',
            'iris manipulation',
            'no label'
        ]

        self._model = None
    
    def _load_model(self):
        filepath = os.path.join(self.model_path, self.model_name)

        try:
            self._model = keras.models.load_model(filepath)
        except:
            print("Unable to load Keras model at {}".format(filepath))
        
    def _load_h5(self, filename):
        filepath = os.path.join(self.h5_path, filename)

        with h5py.File(filepath, 'r', swmr=True) as file:
            X = np.array(file["frames"])
        
        return X
    
    def _standardize(self, X):
        num_pixels = self.height * self.width

        green_channel_mean = sum(X[:, :, :, 0]) / num_pixels
        red_channel_mean = sum(X[:, :, :, 1]) / num_pixels
        blue_channel_mean = sum(X[:, :, :, 2]) / num_pixels

        mean = np.array([green_channel_mean, red_channel_mean, blue_channel_mean])
        X = X[:, 0:256, 100:356, :] - mean

        return X
    
    def _model_summary(self, pretty_print=True):
        if not self._model:
            self._load_model()

        summary = self._model.summary()

        if pretty_print:
            print(summary)
        
        return summary
    
    def predict(self, filename):
        if not self._model:
            self._load_model()
        
        # load in the h5 file and standardize it by mean
        X = self._load_h5(filename)
        X = self._standardize(X)

        # compute the number of model runs necessary 
        # since model can only accept 90 frames at a time
        num_frames = len(X)
        num_batches = num_frames // self.fps + 1

        preds = [] # should be the same length as num_batches = num_seconds of video
        for i in range(num_batches):
            try:
                X_i = np.expand_dims(X[self.fps * i:self.fps * (i+1), :, :, :], axis=0)
                probs_i = self._model.predict_on_batch(X_i)
                preds_i = list(np.argmax(probs_i[0], axis=1))
                most_freq_pred = max(set(preds_i), key = preds_i.count)

                preds.append(self.label_names[most_freq_pred])
                
            except:
                print("Error encountered running model on batch {} with values {}"
                       .format(i, X[self.fps * i:self.fps * (i+1), :, :, :]))
                
                # Append 'no label' for this case
                preds.append(self.label_names[-1])
        
        return preds

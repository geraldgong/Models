import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


class BinaryTransfer:
    def __init__(self):
        self.pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                             include_top=False,
                                             weights='imagenet')

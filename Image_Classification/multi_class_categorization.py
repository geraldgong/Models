import csv
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CNN_multiply:
    def __init__(self):
        self.training_images, self.training_labels = self.load_data_from_csv(f"../Data/MNIST/mnist_train.csv", 28)
        self.testing_images, self.testing_labels = self.load_data_from_csv(f"../Data/MNIST/mnist_test.csv", 28)

    @staticmethod
    def load_data_from_csv(filename, image_width):
        """
        Load images and labels from csv file
        """
        with open(filename) as training_file:
            labels, images = [], []
            for line in training_file.readlines()[1:]:
                line = line.strip().split(',')
                labels.append(line[0])
                images.append(np.array(line[1:]).astype(int).reshape(image_width, -1))

            labels = np.array(labels).astype(int)
            images = np.array(images)
            # Add another dimension to the data (batch_size, height, width, channels)
            images = np.expand_dims(images, axis=3)

        return images, labels

    def image_generator(self):
        # Create an ImageDataGenerator and do Image Augmentation
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        train_generator = train_datagen.flow(self.training_images, self.training_labels)
        validation_datagen = ImageDataGenerator(rescale=1. / 255)
        validation_generator = validation_datagen.flow(self.testing_images, self.testing_labels)

        return train_generator, validation_generator

    def build_model(self, num_classes):
        """
        Sequential model
        """
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3),
                                   activation='relu',
                                   input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        # Compile Model.
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train_model(self, save_model=True):
        """
        Train the Model
        """
        train_generator, validation_generator = self.image_generator()
        callbacks = myCallback()
        history = self.model.fit(
            train_generator,
            epochs=100,
            callbacks=[callbacks],
            validation_data=validation_generator)

        if save_model:
            self.model.save("CNN_multi_classify.h5")
            print(f"Model saved")


class myCallback(tf.keras.callbacks.Callback):
    """
    Define a Callback class that stops training once accuracy reaches 97.0%
    """

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.97:
            print("\nReached 97.0% accuracy so cancelling training!")
            self.model.stop_training = True


if __name__ == "__main__":
    cnn = CNN_multiply()
    cnn.build_model(num_classes=25)
    cnn.train_model()

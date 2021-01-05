import tensorflow as tf
import zipfile
from os import getcwd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


class myCallback(tf.keras.callbacks.Callback):
    """
    Cancels training upon hitting training accuracy of >.999
    """

    def on_epoch_end(self, epoch, logs={}):
        DESIRED_ACCURACY = 0.999
        if logs.get('accuracy') > DESIRED_ACCURACY:
            print('\nDesired accuracy achieved!')
            self.model.stop_training = True


class CNN_binary:
    def __init__(self):
        self.data_path = f"{getcwd()}/../Data/"
        self.width = 150
        self.height = 150
        self.channel = 3

    def unzip_data(self, filename, out_dir):
        zip_ref = zipfile.ZipFile(self.data_path + filename, 'r')
        zip_ref.extractall(self.data_path + out_dir)
        zip_ref.close()

    def build_model(self):
        """
        Sequential model
        """
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(self.width, self.height, self.channel)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        # Compile Model
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=RMSprop(lr=0.001),
            metrics=['accuracy']
        )

    def train_model(self, save_model=True):
        """
        Train the Model
        """
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            self.data_path + "/h-or-s",
            target_size=(self.width, self.height),
            batch_size=16,
            class_mode='binary'
        )

        callbacks = myCallback()

        history = self.model.fit_generator(
            # Your Code Here
            train_generator,
            steps_per_epoch=5,
            epochs=30,
            callbacks=[callbacks]
        )

        if save_model:
            self.model.save("CNN_binary_happy_sad.h5")
            print(f"Model saved")

        return history.history['accuracy'][-1]


if __name__ == "__main__":
    cnn_bi = CNN_binary()
    cnn_bi.unzip_data('happy-or-sad.zip', 'h-or-s')
    cnn_bi.build_model()
    cnn_bi.train_model()
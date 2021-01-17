import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop


class BinaryTransfer:
    def __init__(self):
        self.width = 150
        self.height = 150
        self.channel = 3
        self.batch_size = 32
        self.pre_trained_model = InceptionV3(input_shape=(self.width, self.height, self.channel),
                                             include_top=False,
                                             weights='imagenet')

        self.train_ds, self.val_ds = self.load_data()

    def load_data(self):
        # Horse and humans from tensorflow_datasets
        (train_ds, val_ds), self.info = tfds.load(name='horses_or_humans',
                                                  split=['train', 'test'],
                                                  shuffle_files=True,
                                                  with_info=True,
                                                  as_supervised=True)

        train_ds = (train_ds
                    .shuffle(1000)
                    .map(self.image_augmentation_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                    .batch(self.batch_size)
                    .prefetch(1)
                    )

        val_ds = val_ds.map(self.image_augmentation_valid).batch(self.batch_size).prefetch(1)

        return train_ds, val_ds

    def image_augmentation_train(self, image, label):
        image = tf.image.resize(image, [self.width, self.height])
        image = (image / 255.0)
        image = tf.image.rot90(image)
        image = tf.image.flip_left_right(image)
        image = tf.image.random_crop(image, [self.width, self.height, self.channel])
        final_image = keras.applications.xception.preprocess_input(image)
        return final_image, label

    def image_augmentation_valid(self, image, label):
        image = tf.image.resize(image, [self.width, self.height])
        image = (image / 255.0)
        final_image = keras.applications.xception.preprocess_input(image)
        return final_image, label

    def transfer_model(self):
        """
        Build model with transfer learning
        """
        # Freeze the weights of the pre-trained layers
        for layer in self.pre_trained_model.layers:
            layer.trainable = False
        # Extract certain layer
        last_layer = self.pre_trained_model.get_layer('mixed7')
        last_output = last_layer.output

        # Flatten the output layer to 1 dimension
        x = layers.Flatten()(last_output)
        # Add a fully connected layer with 1,024 hidden units and ReLU activation
        x = layers.Dense(1024, activation='relu')(x)
        # Add a dropout rate of 0.2
        x = layers.Dropout(0.2)(x)
        # Add a final sigmoid layer for classification
        x = layers.Dense(1, activation='sigmoid')(x)

        self.model = Model(self.pre_trained_model.input, x)

        self.model.compile(optimizer=RMSprop(lr=0.0001),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def train_model(self, save_model=True):
        callbacks = myCallback()
        history = self.model.fit(self.train_ds,
                                 epochs=200,
                                 callbacks=[callbacks],
                                 validation_data=self.val_ds)

        if save_model:
            self.model.save("Transfer_binary_classify.h5")
            print(f"Model saved")

        return history


class myCallback(tf.keras.callbacks.Callback):
    """
    Define a Callback class that stops training once accuracy reaches 97.0%
    """

    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.97):
            print("\nReached 97.0% accuracy so cancelling training!")
            self.model.stop_training = True


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))
    trans_model = BinaryTransfer()
    trans_model.transfer_model()
    _ = trans_model.train_model()

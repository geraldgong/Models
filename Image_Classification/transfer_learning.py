import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from tensorflow.keras import Model
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

        self.train_ds, self.val_ds, self.info = self.load_data()

    @staticmethod
    def load_data():
        # Horse and humans from tensorflow_datasets
        (train_ds, val_ds), info = tfds.load(name='horses_or_humans',
                                             split=['train', 'test'],
                                             shuffle_files=True,
                                             with_info=True,
                                             as_supervised=True)

        return train_ds, val_ds, info

    def preprocess_data(self, ds, shuffle=False, augment=False):

        resize_and_rescale = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(self.width, self.height),
            layers.experimental.preprocessing.Rescaling(1. / 255)
        ])

        data_augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.4),
            layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
            layers.experimental.preprocessing.RandomZoom(0.2, 0.2),
        ])

        # Resize and rescale all datasets
        ds = ds.map(lambda x, y: (resize_and_rescale(x), y))
        if shuffle:
            ds = ds.shuffle(1000)

        # Batch all datasets
        ds = ds.batch(self.batch_size)

        # Use data augmentation only on the training set
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))

        # Prefecting on all datasets
        return ds.prefetch(1)

    def build_model(self):
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

    def train_model(self, train_ds, val_ds, save_model=True):
        callbacks = myCallback()
        history = self.model.fit(train_ds,
                                 epochs=200,
                                 callbacks=[callbacks],
                                 validation_data=val_ds)

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
    # Image augumentation on training data
    aug_train_ds = trans_model.preprocess_data(trans_model.train_ds, shuffle=True, augment=True)
    aug_val_ds = trans_model.preprocess_data(trans_model.val_ds, shuffle=True)

    trans_model.build_model()
    _ = trans_model.train_model(aug_train_ds, aug_val_ds)

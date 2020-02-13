from model import MyModel

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils.vis_utils import plot_model

import os


print(tf.__version__)
print(tf.executing_eagerly())


def load_dataset():
    # load fashoion mnist dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # print their shape
    print("x_train shape : {}, y_train shape : {}".format(
        x_train.shape, y_train.shape))
    print("x_test shape : {}, y_test shape : {}".format(
        x_test.shape, y_test.shape))

    # Normalize the data in range [0,1]
    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)
    x_train = tf.reshape(x_train, [-1, 28, 28, 1])
    x_test = tf.reshape(x_test, [-1, 28, 28, 1])

    # one hot encoding
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    return (x_train, x_test, y_train, y_test)


def plot_digits():
    # plot some of digits
    (x_train, x_test, y_train, y_test) = load_dataset()
    fig = plt.figure(figsize=(22, 14))
    ax1 = fig.add_subplot(331)
    plt.imshow(x_train[0][:, :, 0], cmap=plt.get_cmap('gray'))
    ax2 = fig.add_subplot(332)
    plt.imshow(x_train[2][:, :, 0], cmap=plt.get_cmap('gray'))
    ax3 = fig.add_subplot(333)
    plt.imshow(x_train[3][:, :, 0], cmap=plt.get_cmap('gray'))
    fig.show()


def data_generator():
    # Apply data augmentation technique on images to introduce diversity in images
    params = {'featurewise_center': False,
              'samplewise_center': False,
              'featurewise_std_normalization': False,
              'samplewise_std_normalization': False,
              'zca_whitening': False,
              'zca_epsilon': 1e-06,
              'rotation_range': 10,
              'width_shift_range': 0.0,
              'height_shift_range': 0.0,
              'shear_range': 0.1,
              'zoom_range': [1.0, 1.0],
              'channel_shift_range': 0.0,
              'fill_mode': 'nearest',
              'cval': 0.0,
              'horizontal_flip': True,
              'vertical_flip': True
              }
    return ImageDataGenerator(**params)


def train_model():
    # Create instance of model
    model = MyModel()
    SGD_OPTIMIZER = SGD(learning_rate=0.01, momentum=0.001, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD_OPTIMIZER, metrics=["accuracy"])

    schedule_lr = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    (x_train, x_test, y_train, y_test) = load_dataset()

    # Call data generator
    datagen = data_generator()
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=60),
                                  epochs=10,
                                  verbose=2,
                                  steps_per_epoch=500,
                                  validation_data=(x_test, y_test),
                                  callbacks=[schedule_lr, reduce_lr])

    if not os.path.exists("fashionClassifier"):
        os.makedirs("fashionClassifier")
        tf.saved_model.save(model, "fashionClassifier")
    else:
        tf.saved_model.save(model, "fashionClassifier")
    

def infer():
    if not os.path.exists("fashionClassifier"):
        print("model does not exist")
    else:
        model = tf.saved_model.load("fashionClassifier")
        print("Printing available graph signatures")
        print(list(model.signatures.keys()))
        infer = model.signatures["serving_default"]
        print("print input structure")
        print(infer.structured_input_signature)
        print("print output structure")
        print(infer.structured_outputs) 


if __name__ == "__main__":
    load_dataset()
    # train_model()
    #infer()

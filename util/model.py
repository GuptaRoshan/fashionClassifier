import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D

tf.keras.backend.clear_session()

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel,self).__init__()
        self.Conv2D_1 = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3),
                         kernel_initializer='he_uniform', activation='relu', input_shape = (28, 28, 1))
        self.Conv2D_2 = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), activation='relu') 
        self.MaxPooling2D_1 = tf.keras.layers.MaxPooling2D(strides=(2,2))
        self.Conv2D_3 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu')
        self.Conv2D_4 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu')
        self.MaxPooling2D_2 = tf.keras.layers.MaxPooling2D(strides=(2,2))
        self.Flatten = tf.keras.layers.Flatten()
        self.Dense_1  = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')
        self.Dropout_1 = tf.keras.layers.Dropout(0.025)
        self.Dense_2  = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')
        self.Dropout_2 = tf.keras.layers.Dropout(0.025)
        self.Dense_3  = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.Conv2D_1(inputs)
        x = self.Conv2D_2(x)
        x = self.MaxPooling2D_1(x)
        x = self.Conv2D_3(x)
        x = self.Conv2D_4(x)
        x = self.MaxPooling2D_2(x)
        x = self.Flatten(x)     
        x = self.Dense_1(x)
        x = self.Dropout_1(x)
        x = self.Dense_2(x)
        x = self.Dropout_2(x)
        x = self.Dense_3(x)
        return x


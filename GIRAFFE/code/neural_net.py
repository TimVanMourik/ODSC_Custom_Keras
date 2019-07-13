'''
Created by the GiraffeTools Tensorflow generator.
Warning, here be dragons.

'''

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import BatchNormalization

# Model
def NeuralNet(shape):
    model = Sequential()

    model.add(Conv2D(
      (3,3),  # kernel_size
      16,  # filters,
      strides=(1, 1),
      padding='same',
      dilation_rate=(1, 1),
      activation='relu',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(MaxPooling2D(
      pool_size=(2, 2),
      padding='same'
    ))

    model.add(Conv2D(
      (3,3),  # kernel_size
      32,  # filters,
      strides=(1, 1),
      padding='same',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(MaxPooling2D(
      pool_size=(2, 2),
      padding='same'
    ))

    model.add(Conv2D(
      (3,3),  # kernel_size
      64,  # filters,
      strides=(1, 1),
      padding='same',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(MaxPooling2D(
      pool_size=(2, 2),
      padding='same'
    ))

    model.add(Conv2D(
      (3,3),  # kernel_size
      128,  # filters,
      strides=(1, 1),
      padding='same',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(MaxPooling2D(
      pool_size=(2, 2),
      padding='same'
    ))

    model.add(GlobalAveragePooling2D(

    ))

    model.add(Dense(
      64,  # units,
      activation='relu',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(BatchNormalization(
      axis=-1,
      momentum=0.99,
      epsilon=0.001,
      center=True,
      scale=True,
      beta_initializer='zeros',
      gamma_initializer='ones',
      moving_mean_initializer='zeros',
      moving_variance_initializer='ones'
    ))

    model.add(Dense(
      2,  # units,
      activation='sigmoid',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    # Returning model
    return model

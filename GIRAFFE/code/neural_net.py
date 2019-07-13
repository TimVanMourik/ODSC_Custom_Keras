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
      #mandatory argument,  # kernel_size
      #mandatory argument,  # filters,
      strides=(1, 1),
      padding='valid',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(MaxPooling2D(
      pool_size=(2, 2),
      padding='valid'
    ))

    model.add(Conv2D(
      #mandatory argument,  # kernel_size
      #mandatory argument,  # filters,
      strides=(1, 1),
      padding='valid',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(MaxPooling2D(
      pool_size=(2, 2),
      padding='valid'
    ))

    model.add(Conv2D(
      #mandatory argument,  # kernel_size
      #mandatory argument,  # filters,
      strides=(1, 1),
      padding='valid',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(MaxPooling2D(
      pool_size=(2, 2),
      padding='valid'
    ))

    model.add(Conv2D(
      #mandatory argument,  # kernel_size
      #mandatory argument,  # filters,
      strides=(1, 1),
      padding='valid',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(MaxPooling2D(
      pool_size=(2, 2),
      padding='valid'
    ))

    model.add(GlobalAveragePooling2D(

    ))

    model.add(Dense(
      #mandatory argument,  # units,
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
      #mandatory argument,  # units,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    # Returning model
    return model

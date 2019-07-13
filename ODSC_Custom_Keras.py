# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:31:54 2019

@author: w10007346
"""

from GIRAFFE.code.neural_net import NeuralNet

input_shape = (178,218,3)
my_model = NeuralNet(input_shape)

# Show a summary of the model. Check the number of trainable parameters
my_model.summary()

# use early stopping to optimally terminate training through callbacks
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# save best model automatically
mc= ModelCheckpoint('yourdirectory/your_model.h5', monitor='val_loss', 
                    mode='min', verbose=1, save_best_only=True)
cb_list=[es,mc]


# compile model 
my_model.compile(optimizer='adam', loss='binary_crossentropy', 
                 metrics=['accuracy'])


from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# set up data generator
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# get batches of training images from the directory
train_generator = data_generator.flow_from_directory(
        'data/train',
        target_size=input_shape,
        batch_size=12,
        class_mode='categorical')

# get batches of validation images from the directory
validation_generator = data_generator.flow_from_directory(
        'data/valid',
        target_size=input_shape,
        batch_size=12,
        class_mode='categorical')


history = my_model.fit_generator(
        train_generator,
        epochs=30,
        steps_per_epoch=2667,
        validation_data=validation_generator,
        validation_steps=667, callbacks=cb_list)



# plot training and validation accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylim([.5,1.1])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("Custom_Keras_ODSC.png", dpi=300)


####### Testing ################################

# load a saved model
from keras.models import load_model
import os
os.chdir('yourdirectory')
saved_model = load_model('Custom_Keras_CNN.h5')

# generate data for test set of images
test_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Pictures/Celeb_sets/test',
        target_size=input_shape,
        batch_size=1,
        class_mode='categorical',
        shuffle=False)

# obtain predicted activation values for the last dense layer
import numpy as np
test_generator.reset()
pred=saved_model.predict_generator(test_generator, verbose=1, steps=1000)
# determine the maximum activation value for each sample
predicted_class_indices=np.argmax(pred,axis=1)

# label each predicted value to correct gender
labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# format file names to simply male or female
filenames=test_generator.filenames
filenz=[0]
for i in range(0,len(filenames)):
    filenz.append(filenames[i].split('\\')[0])
filenz=filenz[1:]

# determine the test set accuracy
match=[]
for i in range(0,len(filenames)):
    match.append(filenz[i]==predictions[i])
match.count(True)/1000


    


import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout 
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


model = Sequential()
#creating the model convolution layers 1
model.add(Conv2D(32,(3,3),input_shape = (64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
##creating the model convolution layers 2
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
##creating the model convolution layers 3
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
##creating the model convolution layers 4
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
##creating the model convolution layers 5
#model.add(Conv2D(32,(3,3),activation='relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening Of Layers
model.add(Flatten())
#Feeding the data to the neutral network
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=8,activation='relu'))
model.add(Dense(units=3,activation='sigmoid'))


model.compile(SGD(lr=0.1,),loss='categorical_crossentropy',metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                      target_size = (64, 64),
                                                      batch_size = 32,
                                                      class_mode = 'categorical')

test_set = train_datagen.flow_from_directory('dataset/test_set',
                                                      target_size = (64, 64),
                                                      batch_size = 32,
                                                      class_mode = 'categorical')

model.fit_generator(training_set,
                         steps_per_epoch = 100,
                         epochs = 2,
                         validation_data = test_set,
                         validation_steps = 50)

model_json = model.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)
    
model.save_weights('model.h5')
print("Training of Model is done")





















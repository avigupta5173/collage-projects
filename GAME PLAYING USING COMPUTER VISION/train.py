from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import keras

from keras import regularizers, optimizers
from keras.layers import Conv2D,Input,Dense,MaxPooling2D,BatchNormalization,ZeroPadding2D,Flatten,Dropout
from keras.models import Model
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


from  keras.callbacks  import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from numpy.random import permutation

train_images = np.load('train_data.npy')
train_labels = np.load('train_label.npy')

train_images=train_data
train_labels= train_label

train_images = np.array(train_images)
train_labels = np.array(train_labels)




mean = np.mean(train_images,axis=(0,1,2))
std = np.std(train_images,axis=(0,1,2))
train_images = (train_images-mean)/(std+1e-7)



mean = np.mean(screen,axis=(0,1))
std = np.std(screen,axis=(0,1))
screen = (screen-mean)/(std+1e-7)

perm  =  permutation(len(train_images))
train_images = train_images[perm]
train_labels = train_labels[perm]


train_images=np.expand_dims(train_images, axis=3)



lr_reducer = ReduceLROnPlateau(factor = np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-6)
csv_logger = CSVLogger('Lenet.csv')
early_stopper = EarlyStopping(min_delta=0.001,patience=30)
model_checkpoint = ModelCheckpoint('Lenet.hdf5',monitor = 'val_loss', verbose = 1,save_best_only=True)






def le_net():
    model = Sequential()
    # first set of CONV => RELU => POOL
    model.add(Convolution2D(10, 5, 5, border_mode="same",
    input_shape=(60, 60,1)))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Convolution2D(25, 5, 5, border_mode="same"))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    
    model.add(Dense(250))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
 
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    
    model.add(Dense(15))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    
    # softmax classifier
    model.add(Dense(2))
    model.add(Dropout(0.2))
    model.add(Activation("softmax"))
        
    return model

model = le_net()
model.summary()



model.compile(loss='categorical_crossentropy',
        optimizer="Adam",
        metrics=['accuracy'])


model.fit(train_images, train_labels,
              batch_size=12,
              epochs=30,
              validation_split=0.1,
              shuffle=True,callbacks=[lr_reducer,csv_logger,early_stopper,model_checkpoint])


# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    
datagen.fit(new_train)

model.fit(train_images, train_labels,
              batch_size=12,
              epochs=30,
              validation_split=0.3,
              shuffle=True,callbacks=[lr_reducer,csv_logger,early_stopper,model_checkpoint])
model.fit_generator(datagen.flow(new_train, new_labels, batch_size=12),
                        steps_per_epoch=new_train.shape[0] // 12,
                        epochs=30,verbose=0,validation_data=(val_images,val_labels))

model.save('last.hdf5')





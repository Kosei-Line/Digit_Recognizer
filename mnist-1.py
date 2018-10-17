"""Setup"""

import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ReduceLROnPlateau  
from sklearn.model_selection import train_test_split

"""adjust dataset"""
# load training & test datasets
train = pd.read_csv("/content/train.csv")
test = pd.read_csv("/content/test.csv")

# pandas to numpy
Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)

del train

# normalize
X_train = X_train/255.0
test = test/255.0

# reshape the data so that the data
# represents (label, img_rows, img_cols, grayscale)
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# one-hot vector as a label
Y_train = to_categorical(Y_train, num_classes=10)

"""define CNN"""
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))

model.add(Activation('softmax'))
model.summary()


# compile model
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# cross validation
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.10, random_state=1220)
# learning rate
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
# data argumentation
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)
train_generator = gen.flow(X_train, Y_train, batch_size=64)
# model training
model.fit_generator(train_generator, epochs=30, validation_data = (X_val, Y_val), verbose=2, steps_per_epoch=X_train.shape[0]/36,
                                       callbacks=[learning_rate_reduction])


""" test and submit to kaggle"""

# model prediction on test data
predictions = model.predict_classes(test, verbose=0)

# make a submission file
submissions = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions})
submissions.to_csv("my_submission.csv", index=False, header=True)
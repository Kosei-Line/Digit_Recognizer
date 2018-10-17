from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import np_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling as pdp


def build_model(classes):
    sequential = Sequential()
    sequential.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1)))
    sequential.add(BatchNormalization(axis=1))
    sequential.add(MaxPool2D(pool_size=(2, 2)))
    sequential.add(Dropout(0.2))
    sequential.add(Flatten())
    sequential.add(Dense(128, activation='relu'))
    sequential.add(Dense(classes, activation='softmax'))
    sequential.compile(loss='categorical_crossentropy',
                       optimizer='sgd',
                       metrics=['accuracy'])
    return sequential


def plot_validation(train, valid):
    plt.plot(train)
    plt.plot(valid)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def main():
    train = pd.read_csv('data/train.csv', header=0).sample(n=5000)
    test = pd.read_csv('data/test.csv', header=0)
    print(train.info())
    print(test.info())
    pdp.ProfileReport(train)

    x_train = train.drop(['label'], axis=1).as_matrix()
    y_train = train['label']
    x_test = test.as_matrix()

    num_classes = np.max(y_train) + 1
    y_train = np_utils.to_categorical(y_train, 10)

    x_train = x_train.astype('float')
    x_test = x_test.astype('float')

    x_train /= 255
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

    print('x train shape: ', x_train.shape)
    print('y train shape: ', y_train.shape)

    model = build_model(num_classes)
    fit = model.fit(x=x_train, y=y_train,
                    batch_size=32,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_valid, y_valid))

    plot_validation(fit.history['acc'], fit.history['val_acc'])
    plot_validation(fit.history['loss'], fit.history['val_loss'])

    loss, acc = model.evaluate(x_valid, y_valid, verbose=0)
    y_pred = model.predict_classes(x_valid, batch_size=32, verbose=1)
    y_pred = np_utils.to_categorical(y_pred, 10)
    print('Test loss: %s, Test acc: %s' % (loss, acc))
    print(classification_report(y_valid, y_pred))

    test_pred = model.predict_classes(x_test, batch_size=32, verbose=1)
    test_pred = pd.DataFrame(test_pred, columns=['Label'])
    test_pred.index = test_pred.index + 1
    test_pred.to_csv('submit.csv')


if __name__ == '__main__':
    main()

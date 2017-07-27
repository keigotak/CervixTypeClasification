import os
import csv
import random
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D, ELU
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.layers import AveragePooling2D, Input, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import SGD

image_size = (64, 64)

def model_like_vgg16():
    input_image = Input(shape=(*image_size, 3))
    base_model = VGG16(input_tensor=input_image, include_top=False)

    for layer in base_model.layers[:-3]:
        layer.trainable = False

    x = base_model.get_layer("block5_conv3").output
#    x = BatchNormalization()(x)
#    x = AveragePooling2D((2, 2))(x)
#    x = BatchNormalization()(x)
#    x = Dropout(0.5)(x)
#    x = Flatten()(x)
#    x = Dense(4096, activation="elu", W_regularizer=l2(0.01))(x)
#    x = BatchNormalization()(x)
#    x = Dropout(0.5)(x)
#    x = Dense(2048, activation="elu", W_regularizer=l2(0.01))(x)
#    x = BatchNormalization()(x)
#    x = Dropout(0.5)(x)
#    x = Dense(2048, activation="elu", W_regularizer=l2(0.01))(x)
#    x = BatchNormalization()(x)
#    x = Dropout(0.5)(x)
    x = Dense(3, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.summary()

    return model

def model_like_xception():
    input_image = Input(shape=(*image_size, 3))
    base_model = Xception(include_top=False, input_tensor=input_image)

    top_model = Sequential()
    top_model.add(Dense(3, activation="softmax"))
    model = Model(input=base_model.input, output=top_model(base_model.output))

#    x = base_model.output
#    x = GlobalAveragePooling2D()(x)
#    x = BatchNormalization()(x)
#    x = Dropout(0.5)(x)
#    x = Dense(500, activation="relu")(x)
#    x = BatchNormalization()(x)
#    x = Dropout(0.5)(x)
#    x = Dense(500, activation="relu")(x)
#    x = BatchNormalization()(x)
#    x = Dropout(0.5)(x)
#    x = Dense(3, activation="softmax")(x)
#    model = Model(inputs=base_model.input, outputs=x)
    model.summary()

    for layer in model.layers[:-1]:
        layer.trainable = False

    return model

def augment_brightness_camera_images(image):
    '''
    :param image: Input image
    :return: output image with reduced brightness
    '''

    # convert to HSV so that its easy to adjust brightness
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image1[:,:,2] = image1[:,:,2]*random_bright

    # convert to RBG again
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            types = []
            for batch_sample in batch_samples:
                image = batch_sample[1]
                type = batch_sample[0]

                images.append(image)
                types.append(type)

            # trim image to only see section with road
            X_train = np.array(images) / 255.0
            y_train = np.array(types)

            from sklearn.preprocessing import LabelBinarizer
            label_binarizer = LabelBinarizer()
            label_binarizer.fit(["Type_1", "Type_2", "Type_3"])
            y_train = label_binarizer.transform(y_train)

            yield X_train, y_train


def train_by_generator():
    dir_data = "I:\\train"
    wfpath = 'weights.{epoch:02d}--{loss:.2f}-{val_loss:.2f}.h5'
    cp_cb = ModelCheckpoint(filepath=wfpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    es_cb = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

    file_names = glob.glob(os.path.join(dir_data, "**", "*.jpg"), recursive=True)
    random.shuffle(file_names)

    if not os.path.exists("trains.pkl"):
        images = []
        types = []
        datas = {}
        for file_name in file_names:
            type = os.path.split(os.path.dirname(file_name))[-1]

            image = cv2.imread(file_name)
            image = cv2.resize(image, image_size)
            image.astype(np.float32)
            datas[file_name] = [type, image]

            images.append(image)
            types.append(type)
            print("{} of {}".format(len(images), len(file_names)))
        with open('trains.pkl', mode='wb') as f:
            pickle.dump(datas, f)

    if os.path.exists("trains.pkl"):
        with open("trains.pkl", mode="rb") as f:
            pkl = pickle.load(f)
            types_and_images = list(pkl.values())

    train_samples, validation_samples = train_test_split(types_and_images, test_size=0.2)

    batch_size =8
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

#    model = model_like_vgg16()
    model = model_like_xception()
    sgd = SGD(lr=0.0045, decay=0.0001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    # Preprocess incoming data, centered around zero with small standard deviation
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                         validation_data=validation_generator, nb_val_samples=len(validation_samples),
                                         nb_epoch=500, verbose=1, callbacks=[cp_cb, es_cb])

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def train_by_once():
    dir_data = "I:\\train"
    wfpath = 'weights.{epoch:02d}--{loss:.2f}-{val_loss:.2f}.h5'
    cp_cb = ModelCheckpoint(filepath=wfpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    es_cb = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

    file_names = glob.glob(os.path.join(dir_data, "**", "*.jpg"), recursive=True)
    random.shuffle(file_names)

    print("read the images")

    if not os.path.exists("trains.pkl"):
        images = []
        types = []
        datas = {}
        for file_name in file_names:
            type = os.path.split(os.path.dirname(file_name))[-1]

            image = cv2.imread(file_name)
            image = cv2.resize(image, image_size)
            image.astype(np.float32)
            datas[file_name] = [type, image]

            images.append(image)
            types.append(type)
            print("{} of {}".format(len(images), len(file_names)))
        with open('trains.pkl', mode='wb') as f:
            pickle.dump(datas, f)

    if os.path.exists("trains.pkl"):
        with open("trains.pkl", mode="rb") as f:
            pkl = pickle.load(f)
            types_and_images = list(pkl.values())

    X_train = np.array([row[1] for row in types_and_images]) / 255.0
    y_train = np.array([row[0] for row in types_and_images])

    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(y_train)
    y_train = label_binarizer.transform(y_train)

    print("build the models")
#    model = model_like_vgg16()
    model = model_like_xception()
    sgd = SGD(decay=0.0001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    print("fit the models")
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=500, callbacks=[cp_cb, es_cb], batch_size=8)

    model.save("model.h5")

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig("training_result.png")
    plt.show()

    plt.hist(y_train, bins=1)
    plt.savefig("input_data.png")
    plt.show()


if __name__ == '__main__':
#    train_by_generator()
    train_by_once()
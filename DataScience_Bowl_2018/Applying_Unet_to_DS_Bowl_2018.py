#dataset: https://www.kaggle.com/c/data-science-bowl-2018

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

import os
import random
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import tensorflow as tf
import matplotlib.pyplot as plt

seed = 42
np.random.seed = seed

IMG_WIDTH_INPUT = 572
IMG_HEIGHT_INPUT = 572
IMG_CHANNELS_INPUT = 1

IMG_WIDTH_OUTPUT = 388
IMG_HEIGHT_OUTPUT = 388
IMG_CHANNELS_OUTPUT = 1

TRAIN_PATH = '/content/gdrive/MyDrive/train_data/'
TEST_PATH = '/content/gdrive/MyDrive/test_data/'

train_ids = next(os.walk(TRAIN_PATH))[1]
#test_ids = next(os.walk(TEST_PATH))[1][:10]

#train dataset
X_train = np.zeros((len(train_ids), IMG_HEIGHT_INPUT, IMG_WIDTH_INPUT, IMG_CHANNELS_INPUT), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT_OUTPUT, IMG_WIDTH_OUTPUT, IMG_CHANNELS_OUTPUT), dtype=np.bool)
print('Resizing train images and masks... \n')
for i, img_name in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + img_name
    img = imread(path + '/images/' + img_name + '.png')[:,:,:IMG_CHANNELS_INPUT]  
    img = resize(img, (IMG_HEIGHT_INPUT, IMG_WIDTH_INPUT), mode='constant', preserve_range=True)
    X_train[i] = img
    mask = np.zeros((IMG_HEIGHT_OUTPUT, IMG_WIDTH_OUTPUT, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT_OUTPUT, IMG_WIDTH_OUTPUT), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)  
    Y_train[i] = mask
print("done \n")

#test dataset
#X_test = np.zeros((len(test_ids), IMG_HEIGHT_INPUT, IMG_WIDTH_INPUT, IMG_CHANNELS_INPUT), dtype=np.uint8)
#print('Resizing test images...')
#for i, img_name in tqdm(enumerate(test_ids), total=len(test_ids)):
#  path = TEST_PATH + img_name
#  img = imread(path + '/images/' + img_name + '.png')[:,:,:IMG_CHANNELS_INPUT]
#  img = resize(img, (IMG_HEIGHT_INPUT, IMG_WIDTH_INPUT), mode='constant', preserve_range=True)
#  X_test[i] = img
#print("done \n")

#input and its transformation
inputs = tf.keras.layers.Input((IMG_HEIGHT_INPUT, IMG_WIDTH_INPUT, IMG_CHANNELS_INPUT))
start = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)

#contracting path
d1 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'valid')(start)
d1 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'valid')(d1)
t12 = tf.keras.layers.MaxPooling2D((2,2), padding = 'valid')(d1)
d2 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'valid')(t12)
d2 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'valid')(d2)
t23 = tf.keras.layers.MaxPooling2D((2,2), padding = 'valid')(d2)
d3 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'valid')(t23)
d3 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'valid')(d3)
t34 = tf.keras.layers.MaxPooling2D((2,2), padding = 'valid')(d3)
d4 = tf.keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'valid')(t34)
d4 = tf.keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'valid')(d4)
t45 = tf.keras.layers.MaxPooling2D((2,2), padding = 'valid')(d4)
d5 = tf.keras.layers.Conv2D(1024, (3,3), activation = 'relu', padding = 'valid')(t45)
d5 = tf.keras.layers.Conv2D(1024, (3,3), activation = 'relu', padding = 'valid')(d5)

#upscaling path
u4 = tf.keras.layers.Conv2DTranspose(512, (2,2), strides = (2,2), padding = 'valid')(d5)
cc44 = tf.keras.layers.Cropping2D(cropping = ((4,4), (4,4)))(d4)
u4 = tf.keras.layers.concatenate([cc44, u4])
u4 = tf.keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'valid')(u4)
u4 = tf.keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'valid')(u4)
u3 = tf.keras.layers.Conv2DTranspose(256, (2,2), strides = (2,2), padding = 'valid')(u4)
cc33 = tf.keras.layers.Cropping2D(cropping = ((16,16), (16,16)))(d3)
u3 = tf.keras.layers.concatenate([cc33, u3])
u3 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'valid')(u3)
u3 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'valid')(u3)
u2 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides = (2,2), padding = 'valid')(u3)
cc22 = tf.keras.layers.Cropping2D(cropping = ((40,40), (40,40)))(d2)
u2 = tf.keras.layers.concatenate([cc22, u2])
u2 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'valid')(u2)
u2 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'valid')(u2)
u1 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides = (2,2), padding = 'valid')(u2)
cc11 = tf.keras.layers.Cropping2D(cropping = ((88,88), (88,88)))(d1)
u1 = tf.keras.layers.concatenate([cc11, u1])
u1 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'valid')(u1)
u1 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'valid')(u1)

#output
outputs = tf.keras.layers.Conv2D(1, (1,1))(u1)

#model
model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
model.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics=['accuracy'])
#model.summary()

results = model.fit(X_train, Y_train,
                    validation_split=0.2,
                    batch_size=50,
                    epochs=25,
                    verbose=1)

loss = results.history['loss']
val_loss = results.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#test_score = model.evaluate(X_test)

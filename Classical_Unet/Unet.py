#Based on the paper: https://arxiv.org/abs/1505.04597

import tensorflow as tf

IMG_HEIGHT = 572
IMG_WIDTH = 572
IMG_CHANNELS = 1

#input and its transformation
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
start = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)

#contracting path
d1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='valid')(start)
d1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='valid')(d1)
t12 = tf.keras.layers.MaxPooling2D((2,2), padding='valid')(d1)
d2 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='valid')(t12)
d2 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='valid')(d2)
t23 = tf.keras.layers.MaxPooling2D((2,2), padding='valid')(d2)
d3 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='valid')(t23)
d3 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='valid')(d3)
t34 = tf.keras.layers.MaxPooling2D((2,2), padding='valid')(d3)
d4 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='valid')(t34)
d4 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='valid')(d4)
t45 = tf.keras.layers.MaxPooling2D((2,2), padding='valid')(d4)
d5 = tf.keras.layers.Conv2D(1024, (3,3), activation='relu', padding='valid')(t45)
d5 = tf.keras.layers.Conv2D(1024, (3,3), activation='relu', padding='valid')(d5)

#upscaling path
u4 = tf.keras.layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding='valid')(d5)
cc44 = tf.keras.layers.Cropping2D(cropping=((4, 4), (4, 4)))(d4)
u4 = tf.keras.layers.concatenate([cc44, u4])
u4 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='valid')(u4)
u4 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='valid')(u4)
u3 = tf.keras.layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='valid')(u4)
cc33 = tf.keras.layers.Cropping2D(cropping=((16, 16), (16, 16)))(d3)
u3 = tf.keras.layers.concatenate([cc33, u3])
u3 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='valid')(u3)
u3 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='valid')(u3)
u2 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='valid')(u3)
cc22 = tf.keras.layers.Cropping2D(cropping=((40, 40), (40, 40)))(d2)
u2 = tf.keras.layers.concatenate([cc22, u2])
u2 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='valid')(u2)
u2 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='valid')(u2)
u1 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='valid')(u2)
cc11 = tf.keras.layers.Cropping2D(cropping=((88, 88), (88, 88)))(d1)
u1 = tf.keras.layers.concatenate([cc11, u1])
u1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='valid')(u1)
u1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='valid')(u1)

#output
outputs = tf.keras.layers.Conv2D(2, (1,1))(u1)

#model
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='SGD', loss='binary_crossentropy')
model.summary()

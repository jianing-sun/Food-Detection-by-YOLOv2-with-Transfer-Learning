import numpy as np
from keras import callbacks
from keras import layers
from keras import losses, optimizers
from keras.applications.inception_v3 import inception_v3
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras_applications.mobilenet_v2 import MobileNetV2
from keras.utils import np_utils


model = load_model('./models/rn34_224_foodornot_3.h5')
print(model.summary())

# gap_model = Model(inputs=model.input, outputs=model.layers[-3].input)
# print(gap_model.summary())
#
# x = gap_model.output
# x = layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=1, padding='same',
#                   use_bias=False, name='last_conv2d')(x)
# x = layers.BatchNormalization(axis=3, scale=False, name='last_bn')(x)
# x = layers.Activation('relu', name='last_activation')(x)
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dropout(0.33)(x)
# x = layers.Dense(2, activation='softmax')(x)
#
# gap_model = Model(inputs=model.input, outputs=x)
# print(gap_model.summary())


# Preprocess function
def preprocess_im(im):
    im = im.astype(np.float32)
    im = inception_v3.preprocess_input(im)
    return im


# Load data
X_train, y_train = np.load('./data/X_train.npy'), np.load('./data/y_train.npy')
X_val, y_val = np.load('./data/X_val.npy'), np.load('./data/y_val.npy')
X_test, y_test = np.load('./data/X_test.npy'), np.load('./data/y_test.npy')
print('X_train.shape:', X_train.shape)
print('y_train.shape:', y_train.shape)
print('X_val.shape:', X_val.shape)
print('y_val.shape:', y_val.shape)
print('X_test.shape:', X_test.shape)
print('y_test.shape:', y_test.shape)

# Preprocess validation
X_val = inception_v3.preprocess_input(X_val.astype(np.float32))

# Preprocess y
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)

for layer in model.layers:
    layer.trainable = True

# Compile the model
loss = losses.categorical_crossentropy
optimizer = optimizers.SGD(lr=0.001, momentum=0.9)
metrics = ['accuracy']
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# Callbacks
m_q = 'val_loss'
model_path = './models/rn34_224_foodornot_3.h5'
check_pt = callbacks.ModelCheckpoint(filepath=model_path, monitor=m_q, save_best_only=True, verbose=1)
early_stop = callbacks.EarlyStopping(patience=3, monitor=m_q, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(patience=0, factor=0.33, monitor=m_q, verbose=1)
callback_list = [check_pt, early_stop, reduce_lr]

# second fine-tuning
# Data Generator
datagen = ImageDataGenerator(horizontal_flip=True,
                             rotation_range=15,
                             fill_mode='reflect',
                             preprocessing_function=preprocess_im)

# Batch size
batch_size = 64

# Fit
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_val, y_val),
                    epochs=99,
                    steps_per_epoch=len(X_train) / batch_size,
                    callbacks=callback_list)
print('GAP Model Train Done!')

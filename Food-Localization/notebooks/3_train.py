import numpy as np
from keras import callbacks
from keras import losses, optimizers
from keras.applications import inception_v3, mobilenet, mobilenetv2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.layers import UpSampling2D


RANDOM_SEED = 43
np.random.seed(RANDOM_SEED)

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


# Preprocess function
def preprocess_im(im):
    im = im.astype(np.float32)
    im = inception_v3.preprocess_input(im)
    return im


# Preprocess validation
X_val = inception_v3.preprocess_input(X_val.astype(np.float32))

# Preprocess y
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)

# ** Make model **
# Base model
# base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(499, 499, 3))
base_model = mobilenetv2.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.33)(x)
x = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)

print(model.summary())

# ** Configuation **
# Freeze layers (Only top trainable)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
loss = losses.categorical_crossentropy
optimizer = optimizers.Adam()
metrics = ['accuracy']
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# Callbacks
m_q = 'val_loss'
model_path = './models/mnv2_224_foodornot_1.h5'
check_pt = callbacks.ModelCheckpoint(filepath=model_path, monitor=m_q, save_best_only=True, verbose=1)
early_stop = callbacks.EarlyStopping(patience=1, monitor=m_q, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(patience=0, factor=0.33, monitor=m_q, verbose=1)
callback_list = [check_pt, early_stop, reduce_lr]

# ** Fit **
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
print('Top Model Train Done.')

# Load the model
model = load_model('./models/mnv2_224_foodornot_1.h5')

# ** Configuation **
# Open layers
# for layer in model.layers[:249]:
#     layer.trainable = False
# for layer in model.layers[249:]:
#     layer.trainable = True
for layer in model.layers:
    layer.trainable = True

# Compile the model
loss = losses.categorical_crossentropy
optimizer = optimizers.SGD(lr=0.001, momentum=0.9)
metrics = ['accuracy']
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# Callbacks
m_q = 'val_loss'
model_path = './models/mnv2_224_foodornot_2.h5'
check_pt = callbacks.ModelCheckpoint(filepath=model_path, monitor=m_q, save_best_only=True, verbose=1)
early_stop = callbacks.EarlyStopping(patience=1, monitor=m_q, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(patience=0, factor=0.33, monitor=m_q, verbose=1)
callback_list = [check_pt, early_stop, reduce_lr]

# ** Fit **
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
print('All Model Train Done.')

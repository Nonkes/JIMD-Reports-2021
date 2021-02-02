"""
Code to train the quantification CNN (qCNN, ResNet50-based)
(see https://doi.org/10.1002/jmd2.12191)

"""

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.callbacks import ModelCheckpoint

# data augmentation
train_datagen = ImageDataGenerator(
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

# import images for training
train_generator = train_datagen.flow_from_directory(
     ' put filepath here ', 
    batch_size=32,
    class_mode='binary',
    target_size=(224,224))
 
validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

# import images for validation 
validation_generator = validation_datagen.flow_from_directory(
     ' put filepath here ',
    shuffle=False,
    class_mode='binary',
    target_size=(224,224))

# adapt ResNet50 for purposes at hand
conv_base = ResNet50(include_top=False,
                     weights='imagenet',input_shape=(224,224,3))
 
for layer in conv_base.layers:
    layer.trainable = True
 
x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
predictions = layers.Dense(2, activation='softmax')(x)
model = Model(conv_base.input, predictions)
 
optimizer = keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# checkpoint - save in case of performance improvement
filepath="VACS_CYTO_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5" 
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch = 167,
    epochs=50,
    validation_data=validation_generator, 
    validation_steps= 1334, callbacks=callbacks_list, verbose=1)




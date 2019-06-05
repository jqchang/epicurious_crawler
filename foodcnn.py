import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(123)  # for reproducibility
DEBUG = True

# Keep only a single checkpoint, the best over test accuracy.
filepath = "./output/food-cnn-model.hdf5"
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

df = pd.read_csv('food_info_cleaned.csv',header=None)
datagen=ImageDataGenerator(rescale=1./255,validation_split=0.15)
train_generator=datagen.flow_from_dataframe(dataframe=df, directory=".",
            x_col=0, y_col=[1,2,3,4], class_mode="raw",
            target_size=(620,413), batch_size=32, subset="training")
valid_generator=datagen.flow_from_dataframe(dataframe=df, directory=".",
            x_col=0, y_col=[1,2,3,4], class_mode="raw",
            target_size=(620,413), batch_size=32, subset="validation")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
                                input_shape=(620,413,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4, activation='linear'))

model.summary()
opt = tf.keras.optimizers.Adam(lr=0.3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=opt)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10,
                    callbacks=[checkpoint])

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score)

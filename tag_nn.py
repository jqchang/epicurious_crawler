import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
try:
    from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
except ModuleNotFoundError:
    from tensorflow.python.keras.callbacks import ModelCheckpoint,LearningRateScheduler


np.random.seed(123)  # for reproducibility

filepath = './output/tag2cals.hdf5'
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')
df = pd.read_csv('tag_vector.csv',header=None)

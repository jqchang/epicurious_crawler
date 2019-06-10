from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd

df = pd.read_csv('food_info_cleaned.csv')
X = df.to_numpy()
N = X.shape[0]
paths = X[:,0]
Y = X[:,1:] # calories, carbohydrates, protein, fat

path_df = pd.DataFrame(paths)
model = VGG16(weights='imagenet', include_top=False, input_shape=(620,413,3))
datagen = ImageDataGenerator()
img_generator = datagen.flow_from_dataframe(dataframe=path_df, directory='.',
                                            x_col=0, class_mode=None,
                                            shuffle=False, target_size=(620,413))
features = model.predict_generator(img_generator, verbose=1)

np.save(open("imgnet.csv",'w'), features)

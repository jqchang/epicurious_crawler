import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from flask import Flask, Response, request, render_template, send_file, redirect

global graph,model
graph = tf.get_default_graph()


np.random.seed(123)  # for reproducibility
DEBUG = True

# Keep only a single checkpoint, the best over test accuracy.
filepath = "./output/food-cnn-model.hdf5"
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

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

print("loading model from disk...")
model.load_weights(filepath)
print("loaded model from disk")

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if request.method == "POST":
        if request.files.get("image"):
            img_str = request.files["image"].read()
            nparr = np.fromstring(img_str, np.uint8)
            img_np = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
            print("checkpoint")
            img_scaled = cv2.resize(img_np,(620,413))
            img_arr = np.reshape(img_scaled,[1,620,413,3])
            print(img_arr.shape)
            with graph.as_default():
                y = model.predict(img_arr)
            print(y)
            # img_bytes = cv2.imencode(".jpg",img_scaled)[1].tostring()
            return str(y)
            # return Response(img_bytes,
            #     mimetype='image/jpeg')
        else:
            return "No file attached"
    else:
        return redirect('/')


if __name__ == "__main__":
    print("Starting web server")
    app.run()

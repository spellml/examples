from base64 import b64decode
from io import BytesIO
import os

from keras.preprocessing.image import img_to_array
from keras import backend as K
from keras.applications.resnet50 import decode_predictions
import numpy as np
from PIL import Image
import tensorflow as tf

from spell.serving import BasePredictor

MODEL_PATH = os.environ.get("MODEL_PATH", "./model.h5")


class Predictor(BasePredictor):
    def __init__(self):
        # Allow GPU memory to be dynamically allocated to allow multiple model server processes
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.get_default_graph()
        # Construct a TensorFlow session to use for the model server
        with self.graph.as_default():
            self.sess = tf.Session(config=config)
            K.set_session(self.sess)
            # Load model into memory
            self.model = tf.keras.models.load_model(MODEL_PATH)

    # Payload should have a "img" key with a base64-encoded image
    def predict(self, payload):
        image = b64decode(payload["img"])
        x = self.transform_image(image)
        predictions = self.do_predict(x)
        return predictions[0]

    def transform_image(self, image):
        # Resize and scale image
        image = Image.open(BytesIO(image))
        img_resized = image.resize((224, 224), Image.ANTIALIAS)
        scaled_img = img_to_array(img_resized).astype("float32") / 255
        # Wrap image as a np.array
        return np.expand_dims(scaled_img, axis=0)

    def do_predict(self, x):
        with self.graph.as_default():
            pred = self.model.predict(x)
        classes = decode_predictions(pred)
        return [y[0][1] for y in classes]


class BatchPredictor(Predictor):
    def prepare(self, payload):
        return b64decode(payload["img"])

    def predict(self, payload):
        imgs = [self.transform_image(data) for data in payload]
        x = np.array(imgs)
        predictions = self.do_predict(x)
        return predictions

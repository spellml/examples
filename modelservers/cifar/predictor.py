import tensorflow
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

from io import BytesIO
from base64 import b64decode

from spell.serving import BasePredictor

categories = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

MODEL_PATH = "/model/model.h5"

class PythonPredictor(BasePredictor):
    def __init__(self):
        # Load model into memory
        self.model = tensorflow.keras.models.load_model(MODEL_PATH)

    # Payload should have a 'img' key with a base64 encoded image
    def predict(self, payload):
        # Resize and scale image
        img = Image.open(BytesIO(b64decode(payload["img"])))
        img_resized = img.resize((32, 32), Image.ANTIALIAS)
        scaled_img = img_to_array(img_resized).astype("float32") / 255

        # Wrap image in a np.array and evaulate via trained model
        x_test = np.array([scaled_img])
        predict_cat_index = self.model.predict_classes(x_test)[0]
        return categories[predict_cat_index]

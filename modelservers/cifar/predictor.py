import tensorflow
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

from io import BytesIO
from base64 import b64decode

from spell.serving import BasePredictor, metrics

categories = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


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
        probabilities = self.model.predict(x_test)[0]
        index = np.argmax(probabilities)
        probability = probabilities[index]
        prediction = categories[index]

        # Send diagnostic information
        print(f"Predicted {prediction} with confidence {probability}") 
        metrics.send_metric("confidence", probability, tag=prediction)

        return prediction

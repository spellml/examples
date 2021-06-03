import pandas as pd

# import os; os.symlink(
#     "/usr/local/lib/python3.7/dist-packages/tensorflow/libtensorflow_framework.so.1",
#     "/usr/local/lib/python3.7/dist-packages/tensorflow/libtensorflow_framework.so.2"
# )
from ludwig.api import LudwigModel

from spell.serving import BasePredictor

class Predictor(BasePredictor):
    def __init__(self):
        self.ludwig_model = LudwigModel.load("/model/")

    def predict(self, payload):
        # Ludwig expects input to be in the shape of the original DataFrame.
        inp = pd.DataFrame(
            {'description': payload['description'], 'points': payload['points']},
            index=[0]
        )
        # Output is a tuple containing a DataFrame entry, which we parse.
        result, _ = ludwig_model.predict(inp)
        out = {'points': result.points_predictions[0]}
        return out

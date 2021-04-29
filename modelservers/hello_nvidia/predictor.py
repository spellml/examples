from spell.serving import BasePredictor

import subprocess


class Predictor(BasePredictor):
    """
    Print out GPU information when the predictor endpoint is hit.
    """
    def __init__(self):
        pass

    def predict(self, payload):
        proc = subprocess.run(["nvidia-smi"], capture_output=True, check=True, text=True)
        return proc.stdout

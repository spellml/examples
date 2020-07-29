from spell.serving import BasePredictor

import subprocess

class Predictor(BasePredictor):
    """
    Print out GPU information when the predictor endpoint is hit.
    """
    def __init__(self):
        self.encoding = 'utf-8'

    def predict(self, payload):
        time_proc = subprocess.run(["date"], capture_output=True, check=True)
        proc = subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return "{}\n{}".format(time_proc.stdout.decode(self.encoding), proc.stdout.decode(self.encoding))


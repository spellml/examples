import argparse
import base64
import json
import requests
import scipy.misc
from six.moves.urllib.parse import urlparse
import sys
import warnings
import numpy as np
import cv2
import tensorflow as tf
import imageio

def is_url(path):
    try:
        parsed = urlparse(path)
        return bool(parsed.scheme and parsed.netloc)
    except ValueError:
        return False

if __name__ == '__main__':
    # Change path here for input image
    img = scipy.misc.imread('images/input/input_italy.jpg', mode="RGB")
    img = scipy.misc.imresize(img, (256, 256, 3)).astype(np.float32)
    instance = [img.tolist()]
    print('Converting image to array RGB representation...')

    data = json.dumps({"instances" : instance})

    # IMPORTANT: Change `auth_token` to the access token of your spell model server
    # Change `server_url to the url of your spell model server`
    auth_token = 'PLACEHOLDER'
    server_url = "www.google.com"

    # Makes POST request

    headers = {'Authorization':'Bearer {}'.format(auth_token)}
    res = requests.post(server_url, data=data, headers=headers)
    res.raise_for_status()
    prediction = res.json()
    response = json.loads(res.text)
    response_string = response["predictions"][0]
    print(response_string)
    print('Success')

    #Change save path here (if you want to)
    scipy.misc.imsave('images/output/output.jpg',response_string)

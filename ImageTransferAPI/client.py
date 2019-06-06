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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--img', help='JPEG image to classify. Must be an URL path.')
    # # parser.add_argument('--server', help='Prediction server URL.')
    # # parser.add_argument('--auth', help='Prediction server Authentication Token')
    # args = parser.parse_args()
    # img=args.img
    # server=args.server
    # auth=args.auth
    # print('Loading JPEG image: {}'.format(img))
    # if is_url(img):
    #     res = requests.get(img, stream=True)
    #     res.raise_for_status()
    #     image = res.content
    # else:
    with open('input_italy.jpg', "rb") as f:
        image = f.read()
    img = scipy.misc.imread('input_italy.jpg', mode="RGB")
    img = scipy.misc.imresize(img, (256, 256, 3)).astype(np.float32)
    print(img.shape)
    # print(list(img))

    # img = list(img)
    # img = [img for x in range(4)]
    # img = np.array(img).astype(np.float32)
    # print(img)
    # tensor = tf.contrib.util.make_tensor_proto(img, shape=(1,256,256,3))
    # encoded_input_str = base64.b64encode(image)
    # input_string = encoded_input_str.decode("utf-8")
    instance = [img.tolist()]
    # instance.append([[[0, 0, 0] for x in range(256)] for y in range(256)])
    # instance.append([[[0, 0, 0] for x in range(256)] for y in range(256)])
    # instance.append([[[0, 0, 0] for x in range(256)] for y in range(256)])
    print(len(instance))
    print(len(instance[0]))
    print(len(instance[0][0]))
    print(len(instance[0][0][0]))


    # print(tensor.dtype)
    data = json.dumps({"instances" : instance})
    # print(data)
    headers = {'Authorization':'Bearer {}'.format("tqQV0zcmOp3D-4XER1vaL6t4MQHTf-5w7UlxyApPwCXPqB28CFdpmgh5L-ToJsRPDDD9kGUBqJ4j2FUze2VE_Xg")}
    print('processing image...')
    # print(data)
    res = requests.post("https://spell.services/spellrun/imgnet/v3/rest/model:predict", data=data, headers=headers)
    print(res)
    res.raise_for_status()
    prediction = res.json()
    # print(prediction)
    response = json.loads(res.text)
    response_string = response["predictions"][0]
    print(response_string)
    scipy.misc.imsave('output1.jpg',response_string)

    # print("Base64 encoded string: " + response_string[:10] + " ... " + response_string[-10:])
    #
    # # Decode bitstring
    # encoded_response_string = response_string.encode("utf-8")
    # response_image = base64.b64decode(encoded_response_string)
    # print("Raw bitstring: " + str(response_image[:10]) + " ... " + str(response_image[-10:]))

    # Save inferred image

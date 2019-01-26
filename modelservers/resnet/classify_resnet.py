import argparse
import base64
import json
import requests
from six.moves.urllib.parse import urlparse


IMAGE_URL = 'https://images.unsplash.com/photo-1526319238109-524eecb9b913?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=800&q=60'
SERVER_URL = 'https://serving.spell.run/trey/resnet/v1/rest/model:predict'
AUTH_TOKEN = 'tamgBdF9jncH3W8qgwIIVfxPJ_P9u3SnmStkGn3lqgg0hUk3qJSyPD2MIsfhwyI0d38DqVC6t_6N-jqqFbvrC5g'
IMAGE_NET_CLS = 'imagenet1000_clsid_to_human.json'


def is_url(path):
    try:
        parsed = urlparse(path)
        return bool(parsed.scheme and parsed.netloc)
    except ValueError:
        return False


def load_jpeg_img(path):
    print('Loading JPEG image: {}'.format(path))
    if is_url(path):
        res = requests.get(path, stream=True)
        res.raise_for_status()
        return res.content
    else:
        with open(path, "rb") as f:
            return f.read()


def predict_img_class(img_bytes, server, auth):
    print('Predicting image class..')
    data = b'{"instances" : [{"b64": "%s"}]}' % base64.b64encode(img_bytes)
    headers = {'Authorization': 'Bearer {}'.format(auth)}
    res = requests.post(server, data=data, headers=headers)
    res.raise_for_status()
    prediction = res.json()['predictions'][0]
    return prediction['classes']


def img_class_to_name(img_class_id):
    json_data = open(IMAGE_NET_CLS).read()
    data = json.loads(json_data)
    return data[str(img_class_id - 1)]


def classify(img, server, auth):
    img_bytes = load_jpeg_img(img)
    img_cls = predict_img_class(img_bytes, server, auth)
    cls_name = img_class_to_name(img_cls)
    print('Classify result: {}'.format(cls_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify image using an ImageNet model.")
    parser.add_argument('--img', default=IMAGE_URL, help='JPEG image to classify. Must be an URL path.')
    parser.add_argument('--server', default=SERVER_URL, help='Prediction server URL.')
    parser.add_argument('--auth', default=AUTH_TOKEN, help='Prediction server Authentication Token')
    args = parser.parse_args()

    classify(args.img, args.server, args.auth)

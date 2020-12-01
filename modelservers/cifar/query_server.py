#!/usr/bin/env python

import argparse
import base64
import requests
import time

parser = argparse.ArgumentParser(description='Call a spell CIFAR10 model server with a picture.')
parser.add_argument('--url', help='URL of spell model server to query')
parser.add_argument('--image-path', default='dog1.jpeg', help='Path to image file to call predictor with.')

args = parser.parse_args()

with open(args.image_path, "rb") as f:
    image_content = base64.b64encode(f.read()).decode('utf-8')

start = time.time()
response = requests.post(args.url, json={ "img": image_content })
print("Answer: {} ({:.0f} ms)".format(response.text, (time.time() - start) * 1000))

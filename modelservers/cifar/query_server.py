import base64
import requests
import time

URL = "YOUR SERVERS PREDICT ENDPOINT"
IMAGE_FILE_PATH = "dog1.jpeg"

with open(IMAGE_FILE_PATH, "rb") as f:
    image_content = base64.b64encode(f.read()).decode('utf-8')

start = time.time()
response = requests.post(URL, json={ "img": image_content })
print("Answer: {} ({:.0f} ms)".format(response.text, (time.time() - start) * 1000))

import base64
import requests
import time

URL = "http://spell-org.spell-org.spell.services/spell-org/cifar/predict"
#"http://ian-local-aws.ian-local-aws.local.spell.services/ian-local-aws/cifar/predict"
#"http://dev-aws-east-1.spell-external.dev.spell.services/spell-external/cifar10/predict"

with open("dog1.jpeg", "rb") as f:
    image_content = base64.b64encode(f.read()).decode('utf-8')

while True:
    start = time.time()
    response = requests.post(URL, json={ "img": image_content })
    print("Answer: {} ({:.0f} ms)".format(response.text, (time.time() - start) * 1000))
    time.sleep(0.001)


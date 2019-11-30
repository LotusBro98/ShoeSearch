import requests
from PIL import Image
from io import BytesIO
import cv2 as cv
import numpy as np
from encoder.model import NET_SIZE
# import tensorflow as tf

if __name__ == '__main__':
    img = Image.open(BytesIO(requests.get("https://static3.kupivip.ru/V0/04/09/58/68/2x.jpg").content))
    img = np.array(img)
    # img = 1 - tf.image.resize_with_pad(1 - img / 255, NET_SIZE, NET_SIZE).numpy()
    # img = np.uint8(img * 255)
    img = cv.resize(img, (NET_SIZE, NET_SIZE))

    _, imgbytes = cv.imencode(".png", img)
    imgbytes = imgbytes.tobytes()
    res = requests.post("http://localhost:8000", data=imgbytes, headers={"content-type": "image/png", "Content-Length": str(len(imgbytes))})
    print(res.text)
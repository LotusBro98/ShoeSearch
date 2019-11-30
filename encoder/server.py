import tensorflow as tf
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
import cv2 as cv
import json
from PIL import Image
import numpy as np
from encoder.model import Encoder

PORT = 8000

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello, world!')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()

        img = Image.open(BytesIO(body))
        img = np.array(img)
        img = img / 255

        desc = encoder.predict(np.expand_dims(img, axis=0))[0]
        desc = desc.tolist()

        self.wfile.write(json.dumps(desc).encode("utf-8"))

encoder = Encoder()
encoder.load_weights("encoder.h5")

httpd = HTTPServer(('localhost', PORT), SimpleHTTPRequestHandler)
httpd.serve_forever()
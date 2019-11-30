import numpy as np
import glob
import os
import sys
import cv2 as cv
import tensorflow as tf
import random
import pymysql
from pymysql.cursors import DictCursor
from PIL import Image
import requests
import asyncio
from aiohttp import ClientSession
from io import BytesIO

import matplotlib.pyplot as plt
# %matplotlib inline

from encoder.model import NET_SIZE

MIN_EXAMPLES = 16
TRAIN_SIZE = 200

"""## Prepare Dataset"""

connection = pymysql.connect(
    host='185.204.3.233',
    user='admin_neko',
    password='nekoneko',
    db='admin_neko',
    charset='utf8mb4',
    cursorclass=DictCursor
)

cursor = connection.cursor()
query = "SELECT * FROM admin_neko.kupivip_img;"
cursor.execute(query)

item_urls = {}

for row in cursor:
  id = row['id']
  img = row['img']
  if id not in item_urls:
    item_urls[id] = []

  item_urls[id].append(img)
cursor.close()

cursor = connection.cursor()
query = "SELECT * FROM admin_neko.kupivip;"
cursor.execute(query)

item_data = {}
names = set()

for row in cursor:
  id = row['id']
  data = row
  names.add(row['name'])
  del data['id']
  item_data[id] = data

connection.close()

keys = list(item_urls.keys())

item_urls_selected = {}

for key in keys:
  if len(item_urls[key]) < MIN_EXAMPLES:
    continue
  # if item_data[key]['name'] != NAME:
  #   continue
  item_urls_selected[key] = item_urls[key]

print(names)
print(len(item_urls_selected))
print(len(item_data))

labels = []
packs = []

async def fetch(url, session):
  async with session.get(url) as response:
    response = await response.read()
    response = BytesIO(response)
    im = Image.open(response)
    im = 1 - np.array(im) / 255
    im = tf.image.resize_with_pad(im, NET_SIZE, NET_SIZE, ).numpy()
    im = 1 - im
    # im = cv.resize(im, (NET_SIZE, NET_SIZE))
    return im

async def load_from_urls_func(urls):
  tasks = []

  async with ClientSession() as session:
    for url in urls:
      task = asyncio.ensure_future(fetch(url, session))
      tasks.append(task)

    responses = await asyncio.gather(*tasks)
    return responses

def load_from_urls(urls):
  loop = asyncio.get_event_loop()
  imgs = loop.run_until_complete(load_from_urls_func(urls))
  return imgs


keys = list(item_urls_selected.keys())
if TRAIN_SIZE < len(keys):
  keys = random.sample(keys, TRAIN_SIZE)
else:
  random.shuffle(keys)
cnt = 0
all_len = min(len(keys), TRAIN_SIZE)
for key in keys:
  imgs = []
  urls = item_urls_selected[key]
  random.shuffle(urls)

  print(urls)

  try:
    imgs = load_from_urls(urls)
  except Exception as e:
    print(e)
    continue

  random.shuffle(imgs)
  imgs = imgs[:MIN_EXAMPLES]
  imgs = np.float32(imgs)
  packs.append(imgs)
  labels.append(key)

  cnt += 1
  sys.stdout.write("\r{} / {}".format(cnt, all_len))
  sys.stdout.flush()

X_train = np.float32(packs)
np.save("X_train", X_train)
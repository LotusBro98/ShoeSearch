# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import csv
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

NET_SIZE = 128
MIN_EXAMPLES = 8
TRAIN_SIZE = 1000
NAME = 'Полусапоги'

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

X_train = np.load("X_train.npy")

"""## Build Model"""

def MobileResBlock(x, t=6):
  Cin = int(x.shape[-1])

  x = tf.keras.layers.Conv2D(Cin * t, (1, 1))(x)
  x = tf.keras.layers.Activation('tanh')(x)
  x = tf.keras.layers.BatchNormalization()(x)

  x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same')(x)
  x = tf.keras.layers.Activation('tanh')(x)
  x = tf.keras.layers.BatchNormalization()(x)

  x = tf.keras.layers.Conv2D(Cin, (1, 1))(x)
  x = tf.keras.layers.Activation('tanh')(x)
  x = tf.keras.layers.BatchNormalization()(x)

  return x

def downsample(x):
  
  # x = x + MobileResBlock(x)
  # x = x + MobileResBlock(x)
  x = x + MobileResBlock(x)

  x = tf.keras.layers.Conv2D(int(x.shape[-1] * 2), (3,3), strides=(2,2), padding='same')(x)
  x = tf.keras.layers.Activation('tanh')(x)
  x = tf.keras.layers.BatchNormalization()(x)

  return x

def Encoder():
  inputs = tf.keras.layers.Input((NET_SIZE, NET_SIZE, 3))

  x = inputs

  for i in range(int(np.log2(NET_SIZE))):
    x = downsample(x)
    print(x.shape)

  x = tf.keras.layers.Flatten()(x)

  x = tf.keras.layers.Dense(128, activation='tanh')(x)

  x = tf.keras.layers.Dense(16, activation='sigmoid')(x)

  return tf.keras.Model(inputs = inputs, outputs = x)

# model = Encoder()

def loss(ds, mds):
  total_loss = 0
  eps = 0.01
  stds = []
  for d in ds:
    std1 = tf.reduce_mean(tf.math.reduce_std(d, axis=-2))
    loss_add = - tf.math.log((1 - std1) * (1 - eps) + eps)
    stds.append(std1)
    total_loss += loss_add

  std = 0
  mds = tf.concat(mds, axis=0)
  for i in range(1, len(ds)):
    stdd = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(mds - tf.roll(mds, i, axis=0)), axis=-1)))
    std += stdd
  std /= (len(ds) - 1)
  # std = tf.reduce_mean(tf.math.reduce_std(tf.concat(ds, axis=0), axis=0))
  loss_add = - 1 * len(ds) * tf.math.log(std * (1 - eps) + eps)
  
  total_loss += loss_add

  return total_loss, np.average(stds), std.numpy()

optimizer = tf.keras.optimizers.Adam(1e-3)

"""## Train"""

def train_step(xs, ep):
  with tf.GradientTape() as tape:
    ds = []
    mds = []
    for x in xs:
      d = model(x)
      ds.append(d)
      mds.append(tf.expand_dims(tf.reduce_mean(d, axis=-2), axis=0))

    # print(ds[0])
    # print(d2)

    loss_tensor, std1, std = loss(ds, mds)
    print("{} {} {} {}".format(ep, loss_tensor.numpy(), std1, std))

    gradients = tape.gradient(loss_tensor, model.trainable_variables)
    # print(gradients)
    if np.isnan(gradients[0][0][0][0][0].numpy()):
      return
    

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

model = Encoder()

for i in range(1000):
  train_step(X_train[random.sample(range(len(X_train)), 2)], i)

model.save("encoder.h5")

model.load_weights("encoder.h5")

diffs = []
for pack in X_train:
  d = model.predict(np.float32(pack))
  diffs.append((d, pack))

SAMPLES = 10
TOP_N = 7

for i in range(SAMPLES):
  fig = plt.figure(figsize=(20, 20))
  axarr = fig.subplots(1,TOP_N + 1)

  i1 = random.randint(0, len(X_train) - 1)
  d1 = random.choice(model.predict(np.float32(X_train[i1])))

  diff = sorted(diffs, key=lambda x: np.average(np.sqrt(np.average(np.square(d1 - x[0]), axis=-1))))

  axarr[0].imshow(X_train[i1][0])
  j = 1
  for d, pack in diff[:TOP_N]:
    axarr[j].imshow(pack[-1])
    j += 1
  plt.show()


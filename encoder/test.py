from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random
from encoder.model import Encoder

model = Encoder()
model.load_weights("encoder.h5")

X_train = np.load("X_train.npy")

diffs = []
for pack in X_train:
  d = model.predict(np.float32(pack))
  diffs.append((d, pack))

SAMPLES = 10
TOP_N = 7

fig = plt.figure(figsize=(20, 20))
axarr = fig.subplots(SAMPLES, TOP_N + 1)
for i in range(SAMPLES):
  i1 = random.randint(0, len(X_train) - 1)
  d1 = random.choice(model.predict(np.float32(X_train[i1])))

  diff = sorted(diffs, key=lambda x: np.average(np.sqrt(np.average(np.square(d1 - x[0]), axis=-1))))

  axarr[i, 0].imshow(X_train[i1][0])
  j = 1
  for d, pack in diff[:TOP_N]:
    axarr[i, j].imshow(pack[-1])
    j += 1
plt.show()


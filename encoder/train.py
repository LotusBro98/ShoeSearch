import tensorflow as tf
import numpy as np
import random
from encoder.model import Encoder

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
X_train = np.load("X_train.npy")

try:
    model.load_weights("encoder.h5")
except Exception as e:
    print("Failed to load weights")


for i in range(1000):
    train_step(X_train[random.sample(range(len(X_train)), 2)], i)
    if i % 10 == 0:
        model.save("encoder.h5")
        print("Saved model")
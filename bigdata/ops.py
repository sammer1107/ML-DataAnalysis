import tensorflow as tf
import numpy as np


def bent_identity(x):
    return tf.add((tf.sqrt(tf.square(x) + 1) - 1) / 2, x, name='bent_identity')


def escalator(x):
    return tf.add(tf.pow(x,3), x*3/4, name='escalator')


def hard_dropout(x, num_drop, axis=-1, training=False):
    if not training:
        return x
    if axis < 0:
        axis = len(x.shape) + axis
    assert axis >= 0

    shape = [int(i) for i in x.shape]
    # length of dropout mask
    length = shape[axis]
    del shape[axis]
    # number of dropout mask (for each example, column and channel)
    num_repeat = np.prod(shape)
    masks = []
    mask = np.concatenate([np.ones(length - num_drop, dtype=np.float32) * length / (length - num_drop),
                           np.zeros(num_drop, dtype=np.float32)])
    with tf.name_scope('hard_dropout'):
        for i in range(num_repeat):
            masks.append(tf.random_shuffle(mask))
        masks = tf.stack(masks, axis=0)
        # reshape and transpose the masks to the shape of input tensor
        masks = tf.reshape(masks, [*shape, length])
        permutation = list(range(0,len(shape)))
        permutation.insert(axis, len(shape))
        masks = tf.transpose(masks, permutation)
        x = masks * x
    return x

# awfully slow
# def hard_dropout_v2(x, num_drop, axis, training=False):
#     if not training:
#         return x
#
#     shape = [int(i) for i in x.shape]
#     num_repeat = int(np.prod(shape)/shape[axis])
#     length = shape[axis]
#     masks = [[0]*num_drop + [length / (length - num_drop)]*(length-num_drop)]*num_repeat
#
#     with tf.name_scope('hard_dropout_v2'):
#         masks = tf.constant(masks, dtype=tf.float32, name='dropout_mask')
#         masks = tf.map_fn(tf.random_shuffle, masks)
#
#         shape[axis], shape[-1] = shape[-1], shape[axis]
#         masks = tf.reshape(masks, shape)
#
#         permutation = list(range(len(shape)))
#         permutation[axis], permutation[-1] = permutation[-1], permutation[axis]
#         x = tf.multiply(tf.transpose(masks, permutation), x)
#
#     return x

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
    length = shape[axis]
    del shape[axis]
    num_repeat = np.prod(shape)
    masks = []
    mask = np.concatenate([np.ones(length - num_drop, dtype=np.float32) * length / (length - num_drop),
                           np.zeros(num_drop, dtype=np.float32)])
    with tf.name_scope('hard_dropout'):
        for i in range(num_repeat):
            masks.append(tf.random_shuffle(mask))
        masks = tf.stack(masks, axis=0)
        masks = tf.reshape(masks, [*shape, length])
        permutation = list(range(0,len(shape)))
        permutation.insert(axis, len(shape))
        masks = tf.transpose(masks, permutation)
        x = masks * x
    return x


# def crop_dropout(x, num_drop, axis=-1, training=False):
#     if not training:
#         return x
#
#     shape = [int(i) for i in x.shape]
#     length = shape[axis]
#     del shape[axis]
#     num_repeat = np.prod(shape)
#     masks = []
#     mask = np.concatenate([np.ones(length - num_drop, dtype=np.float32),
#                            np.zeros(num_drop, dtype=np.float32)])
#     mask = mask.astype(np.bool)
#     with tf.name_scope('hard_dropout'):
#         for i in range(num_repeat):
#             masks.append(tf.random_shuffle(mask))
#         masks = tf.stack(masks, axis=0)
#         masks = tf.reshape(masks, [*shape, length])
#         permutation = list(range(0, len(shape)))
#         permutation.insert(axis, len(shape))
#         masks = tf.transpose(masks, permutation)
#         x = tf.boolean_mask(x, masks)
#     print(x.shape)
#     return x

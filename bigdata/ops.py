import tensorflow as tf


def bent_identity(x):
    with tf.variable_scope('bent_identity'):
        return (tf.sqrt(tf.square(x) + 1) - 1) / 2 + x



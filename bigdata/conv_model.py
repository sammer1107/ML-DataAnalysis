import tensorflow as tf
from bigdata.data.thu_dataset import ThuDataset


def conv_model(inputs, targets, learning_rate, mode='train', save_summary=False):
    """a simple convolution model for Thu Dataset,
    the input should be in shape [batch_size, 7500, 4, 1]
    preprocessed from thu_dataset.py """

    assert mode in ['train', 'eval'], "mode should be one of ['train', 'eval']"

    fetches = {}
    net = tf.layers.conv2d(inputs, filters=8,
                           kernel_size=[10, 4],
                           activation=tf.nn.leaky_relu,
                           name='conv1')
    net = tf.squeeze(tf.reduce_mean(net, axis=1), axis=1)
    net = tf.layers.dense(net, units=5, name='dense1', activation=tf.nn.leaky_relu)
    net = tf.layers.dense(net, units=1, name='output')
    net = tf.squeeze(net, axis=1)
    fetches['outputs'] = net
    fetches['relative_error'] = (net - targets) / targets
    fetches['squared_error'] = tf.square(net - targets)
    loss = tf.losses.mean_squared_error(net, targets)
    fetches['loss'] = loss

    if mode == 'train':
        global_step = tf.Variable(0, trainable=False)
        opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.8)
        train_op = opt.minimize(loss, global_step=global_step)
        fetches['global_step'] = global_step
        fetches['train_op'] = train_op

        if save_summary:
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('loss', loss)
            with tf.variable_scope('conv1', reuse=True):
                kernel = tf.get_variable('kernel')
            tf.summary.image('kernels', tf.transpose(kernel, [3,0,1,2]), max_outputs=8)
            fetches['summary_all'] = tf.summary.merge_all()
    return fetches


def pooled_conv_model(inputs, targets, learning_rate, learning_rate_decay=0.97, mode='train', save_summary=False):
    """This is a modified experimental model for Thu Dataset,
    this uses a big pooling first in order to simplify the feature
    the input should be in shape [batch_size, 7500, 4, 1]
    preprocessed from thu_dataset.py """

    assert mode in ['train', 'eval'], "mode should be one of ['train', 'eval']"

    fetches = {}

    net = tf.layers.average_pooling2d(inputs, pool_size=[375,1],
                                      strides=[375,1],
                                      name='pooling')

    net = tf.layers.conv2d(net, filters=7,
                           kernel_size=[4,4],
                           strides=[4,1],
                           name='conv1',
                           activation=tf.nn.tanh)
    net = tf.squeeze(net, axis=[2])

    net = tf.layers.conv1d(net, filters=1,
                           kernel_size=5,
                           name='conv2',
                           activation=tf.nn.tanh)
    net = tf.squeeze(net, axis=[1])

    outputs = tf.layers.dense(net, 1, name='output')
    outputs = tf.reshape(outputs,[-1])

    fetches['outputs'] = outputs
    fetches['relative_error'] = (outputs - targets) / targets
    loss = tf.losses.mean_squared_error(outputs, targets)
    # loss = tf.reduce_mean(tf.abs(outputs - targets))
    fetches['loss'] = loss

    if mode == 'train':
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=10000,
                                                   decay_rate=learning_rate_decay,
                                                   name='decayed_learning_rate',
                                                   staircase=True)
        opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.8)
        train_op = opt.minimize(loss, global_step=global_step)
        fetches['global_step'] = global_step
        fetches['train_op'] = train_op

        if save_summary:
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('loss', loss)
            with tf.variable_scope('conv1', reuse=True):
                kernel = tf.get_variable('kernel')
                tf.summary.image('kernels', tf.transpose(kernel, [3,0,1,2]), max_outputs=10)
            fetches['summary_all'] = tf.summary.merge_all()

    return fetches


def pooled_conv_model_m(inputs, targets, learning_rate, learning_rate_decay=0.97, mode='train', save_summary=False):
    """This is a modified experimental model for Thu Dataset,
    this uses a big pooling first in order to simplify the feature
    the input should be in shape [batch_size, 7500, 4, 1]
    preprocessed from thu_dataset.py,
    The model is modified from pooled_conv_model and changed the stride of conv2d to 1"""

    assert mode in ['train', 'eval'], "mode should be one of ['train', 'eval']"

    fetches = {}

    net = tf.layers.average_pooling2d(inputs, pool_size=[375,1],
                                      strides=[375,1],
                                      name='pooling')

    net = tf.layers.conv2d(net, filters=2,
                           kernel_size=[4,4],
                           strides=[1,1],
                           name='conv1',
                           activation=tf.nn.leaky_relu)
    net = tf.squeeze(net, axis=[2])

    net = tf.layers.conv1d(net, filters=1,
                           kernel_size=3,
                           name='conv2',
                           activation=tf.nn.leaky_relu)
    net = tf.reduce_mean(net, axis=1, name='mean')

    outputs = tf.layers.dense(net, 1, name='output')
    outputs = tf.reshape(outputs,[-1])

    fetches['outputs'] = outputs
    fetches['relative_error'] = (outputs - targets) / targets
    fetches['squared_error'] = tf.square(outputs - targets)
    loss = tf.losses.mean_squared_error(outputs, targets)
    fetches['loss'] = loss

    if mode == 'train':
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=10000,
                                                   decay_rate=learning_rate_decay,
                                                   name='decayed_learning_rate',
                                                   staircase=True)
        opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.8)
        train_op = opt.minimize(loss, global_step=global_step)
        fetches['global_step'] = global_step
        fetches['train_op'] = train_op

        if save_summary:
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('loss', loss)
            with tf.variable_scope('conv1', reuse=True):
                kernel = tf.get_variable('kernel')
                tf.summary.image('kernels', tf.transpose(kernel, [3,0,1,2]), max_outputs=5)
            fetches['summary_all'] = tf.summary.merge_all()

    return fetches


def pooled_conv2d_model(inputs, targets, learning_rate, learning_rate_decay=0.97, mode='train', save_summary=False):
    """This is a new experimental model for Thu Dataset,
    this also uses a big pooling first in order to simplify the feature.
    the input should be in shape [batch_size, 7500, 4, 1] preprocessed from thu_dataset.py.
    Model Architecture:
    pooling : [batch, 7500, 4, 1]       => [batch, 20, 4, 1]
    conv2d  : [batch, 20, 4, 1]         => [batch, 5, 4, channels]
    conv2d  : [batch, 5, 4, channels]   => [batch, 1, 1, channels]
    squeeze : [batch, 1, 1, channels]   => [batch, channels]
    dense+reshape : [batch, channels]   => [batch] """

    assert mode in ['train', 'eval'], "mode should be one of ['train', 'eval']"

    fetches = {}

    net = tf.layers.average_pooling2d(inputs, pool_size=[375, 1],
                                      strides=[375, 1],
                                      name='pooling')
    if save_summary:
        tf.summary.image('AvgPooling', net, family='outputs')

    net = tf.layers.conv2d(net, filters=5,
                           kernel_size=[4,1],
                           strides=[4,1],
                           name='conv1',
                           activation=tf.nn.tanh)
    if save_summary:
        with tf.variable_scope('conv1', reuse=True):
            filter = tf.get_variable('kernel')
        deconv = tf.nn.conv2d_transpose(net, filter, [32,20,4,1], strides=[1,4,1,1])
        tf.summary.image('conv1', deconv, family='outputs')

    net = tf.layers.conv2d(net, filters=1,
                           kernel_size=[5,4],
                           name='conv2',
                           activation=tf.nn.tanh)

    net = tf.squeeze(net, axis=[1,2])
    outputs = tf.layers.dense(net, 1, name='output')
    outputs = tf.reshape(outputs,[-1])

    fetches['outputs'] = outputs
    fetches['relative_error'] = (outputs - targets) / targets
    loss = tf.losses.mean_squared_error(outputs, targets)
    # loss = tf.reduce_mean(tf.abs(outputs - targets))
    fetches['loss'] = loss

    if mode == 'train':
        global_step = tf.Variable(0, trainable=False)  # , name='global_step')
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=10000,
                                                   decay_rate=learning_rate_decay,
                                                   name='decayed_learning_rate',
                                                   staircase=True)
        opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        train_op = opt.minimize(loss, global_step=global_step)
        fetches['global_step'] = global_step
        fetches['train_op'] = train_op

        if save_summary:
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('loss', loss)
            # conv1
            with tf.variable_scope('conv1', reuse=True):
                kernel = tf.get_variable('kernel')
                kernel = tf.transpose(kernel, [3,0,1,2])
                kernel = tf.concat(tf.unstack(kernel, axis=0),axis=1)
                kernel = tf.expand_dims(kernel, axis=0)
            tf.summary.image('kernels', kernel, max_outputs=10, family='conv1')
            # conv2
            with tf.variable_scope('conv2', reuse=True):
                kernels = tf.get_variable('kernel')
            for i, kernel in enumerate(tf.unstack(kernels, axis=3)):
                kernel = tf.transpose(kernel, [2,0,1])
                kernel = tf.expand_dims(kernel, axis=3)
                tf.summary.image('kernel_{}'.format(i), kernel, max_outputs=10, family='conv2')
            fetches['summary_all'] = tf.summary.merge_all()

    return fetches


def pooled_conv2d_model_250(inputs, targets, learning_rate, learning_rate_decay=0.97, mode='train', save_summary=False):
    """This is a new experimental model for Thu Dataset,
    this also uses a big pooling first in order to simplify the feature.
    the input should be in shape [batch_size, 7500, 4, 1] preprocessed from thu_dataset.py.
    Model Architecture:
    pooling : [batch, 7500, 4, 1]       => [batch, 30, 4, 1]
    conv2d  : [batch, 30, 4, 1]         => [batch, 10, 4, channels]
    conv2d  : [batch, 10, 4, channels]  => [batch, 5, 4, channels]
    conv2d  : [batch, 5, 4, channels]   => [batch, 1, 1, channels]
    squeeze : [batch, 1, 1, channels]   => [batch, channels]
    dense+reshape : [batch, channels]   => [batch] """

    assert mode in ['train', 'eval'], "mode should be one of ['train', 'eval']"

    fetches = {}
    # if mode == 'train':
    #     inputs, targets = tf.train.shuffle_batch([inputs, targets], batch_size=8, capacity=320,
    #                                              min_after_dequeue=160,
    #                                              enqueue_many=True,
    #                                              name='batch')

    net = tf.layers.average_pooling2d(inputs, pool_size=[250, 1],
                                      strides=[250, 1],
                                      name='pooling')

    net = tf.layers.conv2d(net, filters=5,
                           kernel_size=[3,1],
                           strides=[3,1],
                           name='conv1',
                           activation=tf.nn.tanh)

    net = tf.layers.conv2d(net, filters=3,
                           kernel_size=[2, 1],
                           strides=[2, 1],
                           name='conv2',
                           activation=tf.nn.tanh)

    net = tf.layers.conv2d(net, filters=1,
                           kernel_size=[5,4],
                           name='conv3',
                           activation=tf.nn.tanh)

    net = tf.squeeze(net, axis=[1,2])
    outputs = tf.layers.dense(net, 1, name='output')
    outputs = tf.reshape(outputs,[-1])

    fetches['outputs'] = outputs
    fetches['relative_error'] = (outputs - targets) / targets
    loss = tf.losses.mean_squared_error(outputs, targets)
    # loss = tf.reduce_mean(tf.abs(outputs - targets))
    fetches['loss'] = loss

    if mode == 'train':
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=10000,
                                                   decay_rate=learning_rate_decay,
                                                   name='decayed_learning_rate',
                                                   staircase=True)
        opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        train_op = opt.minimize(loss, global_step=global_step)
        fetches['global_step'] = global_step
        fetches['train_op'] = train_op

        if save_summary:
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('loss', loss)
            # conv1 summary
            # kernels are in shape [H,W,C,N]
            with tf.variable_scope('conv1', reuse=True):
                kernel = tf.get_variable('kernel')
                # kernel = tf.transpose(kernel, [3,0,1,2])
                kernel = tf.concat(tf.unstack(kernel, axis=3),axis=1)
                kernel = tf.expand_dims(kernel, axis=0)
            tf.summary.image('kernels', kernel, max_outputs=10, family='conv1')
            # conv2 summary
            with tf.variable_scope('conv2', reuse=True):
                kernel = tf.get_variable('kernel')
                kernel = tf.transpose(kernel, [2,0,1,3])
                kernel = tf.concat(tf.unstack(kernel, axis=3), axis=2)
                kernel = tf.expand_dims(kernel, axis=3)
            tf.summary.image('kernels', kernel, max_outputs=10, family='conv2')
            # conv3 summary
            with tf.variable_scope('conv3', reuse=True):
                kernels = tf.get_variable('kernel')
            for i, kernel in enumerate(tf.unstack(kernels, axis=3)):
                kernel = tf.transpose(kernel, [2,0,1])
                kernel = tf.expand_dims(kernel, axis=3)
                tf.summary.image('kernel_{}'.format(i), kernel, max_outputs=3, family='conv3')
            fetches['summary_all'] = tf.summary.merge_all()

    return fetches


def pooled_conv2d_model_2506(inputs, targets, learning_rate, learning_rate_decay=0.97, mode='train', save_summary=False):
    """This is a new experimental model for Thu Dataset,
    this also uses a big pooling first in order to simplify the feature.
    the input should be in shape [batch_size, 7500, 4, 1] preprocessed from thu_dataset.py.
    Model Architecture:
    pooling : [batch, 7500, 4, 1]       => [batch, 30, 4, 1]
    conv2d  : [batch, 30, 4, 1]         => [batch, 5, 4, channels]
    conv2d  : [batch, 5, 4, channels]   => [batch, 1, 1, channels]
    squeeze : [batch, 1, 1, channels]   => [batch, channels]
    dense+reshape : [batch, channels]   => [batch] """

    assert mode in ['train', 'eval'], "mode should be one of ['train', 'eval']"

    fetches = {}

    net = tf.layers.average_pooling2d(inputs, pool_size=[250, 1],
                                      strides=[250, 1],
                                      name='pooling')
    if save_summary:
        tf.summary.image('AvgPooling', net, family='outputs')

    net = tf.layers.conv2d(net, filters=5,
                           kernel_size=[3,1],
                           strides=[3,1],
                           name='conv1',
                           activation=tf.nn.tanh)
    if save_summary:
        with tf.variable_scope('conv1', reuse=True):
            filter = tf.get_variable('kernel')
        deconv = tf.nn.conv2d_transpose(net, filter, [32,30,4,1], strides=[1,3,1,1])
        tf.summary.image('conv1', deconv, family='outputs')

    net = tf.layers.conv2d(net, filters=3,
                           kernel_size=[1,1],
                           name='conv2',
                           activation=tf.nn.tanh)

    net = tf.layers.conv2d(net, filters=2,
                           kernel_size=[5,4],
                           dilation_rate=[2,1],
                           activation=tf.nn.tanh,
                           name='conv3')

    net = tf.layers.flatten(net, name='flatten')

    # net = tf.squeeze(net, axis=[1,2])

    outputs = tf.layers.dense(net, 1, name='output')
    outputs = tf.reshape(outputs,[-1])

    fetches['outputs'] = outputs
    fetches['relative_error'] = (outputs - targets) / targets
    loss = tf.losses.mean_squared_error(outputs, targets)
    # loss = tf.reduce_mean(tf.abs(outputs - targets))
    fetches['loss'] = loss

    if mode == 'train':
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=10000,
                                                   decay_rate=learning_rate_decay,
                                                   name='decayed_learning_rate',
                                                   staircase=True)
        opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        train_op = opt.minimize(loss, global_step=global_step)
        fetches['global_step'] = global_step
        fetches['train_op'] = train_op

        if save_summary:
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('loss', loss)
            # conv1 summary
            # kernels are in shape [H,W,C,N]
            with tf.variable_scope('conv1', reuse=True):
                kernel = tf.get_variable('kernel')
                # kernel = tf.transpose(kernel, [3,0,1,2])
                kernel = tf.concat(tf.unstack(kernel, axis=3),axis=1)
                kernel = tf.expand_dims(kernel, axis=0)
            tf.summary.image('kernels', kernel, max_outputs=10, family='conv1')
            # conv2 summary
            # with tf.variable_scope('conv2', reuse=True):
            #     kernel = tf.get_variable('kernel')
            #     kernel = tf.transpose(kernel, [2,0,1,3])
            #     kernel = tf.concat(tf.unstack(kernel, axis=3), axis=2)
            #     kernel = tf.expand_dims(kernel, axis=3)
            # tf.summary.image('kernels', kernel, max_outputs=10, family='conv2')

            fetches['summary_all'] = tf.summary.merge_all()

    return fetches


# from bigdata.data.thu_dataset import ThuDataset
# dataset = ThuDataset('bigdata/log_normalized_data/{}/')
# inputs, targets = dataset.get_data()
# inputs = tf.constant(inputs, dtype=tf.float32)
# targets = tf.constant(targets, dtype=tf.float32)
# fetch = pooled_conv2d_model(inputs, targets, 0.001, save_summary=True)
#
# with tf.Session() as sess:
#     print(sess.run('conv2_1/shape:0'))

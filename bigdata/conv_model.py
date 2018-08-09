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


import tensorflow as tf
from bigdata.data.thu_dataset import ThuDataset


def conv_model(inputs, targets, learning_rate, mode='train', save_summary=False):
    """a simple convolution model for Thu Dataset,
    the input should be in shape [batch_size, 7500, 4, 1]
    preprocessed from thu_dataset.py """

    assert mode in ['train', 'eval'], "mode should be one of ['train', 'eval']"

    fetches = {}

    inputs = linear_per_column(inputs)

    net = tf.layers.conv2d(inputs, filters=1,
                           kernel_size=[1, 4],
                           strides=[1,1],
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
            graph = tf.get_default_graph()
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('loss', loss)
            # kernel visualization
            with tf.variable_scope('conv1', reuse=True):
                kernel = tf.get_variable('kernel')
            tf.summary.image('kernels', tf.transpose(kernel, [3,0,1,2]), max_outputs=8)
            # linear_per_column visualization
            for var in ['bias']:
                tensor = graph.get_tensor_by_name('linear_per_column/'+var+':0')
                tf.summary.histogram(tensor.name, tensor)
            fetches['summary_all'] = tf.summary.merge_all()
    return fetches


def linear_per_column(inputs):
    biases = []
    for i in range(4):  # for each column
        with tf.variable_scope("linear_{}".format(i)):
            # weights.append(tf.get_variable(name='weight', shape=1, initializer=tf.initializers.constant(1)))
            biases.append(tf.get_variable(name='bias', shape=1,
                                          initializer=tf.initializers.constant(-0.0101),
                                          trainable=True))

    with tf.variable_scope('linear_per_column'):
        # tf.identity(weights, name='weight')
        tf.identity(biases, name='bias')

        # weights = tf.tile(tf.reshape(weights, [1, 4, 1]), [7500, 1, 1])
        biases = tf.tile(tf.reshape(biases, [1, 4, 1]), [7500, 1, 1])

        inputs = inputs*tf.cast((inputs+biases) > 0, tf.float32)

    return inputs

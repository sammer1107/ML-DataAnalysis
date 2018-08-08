import tensorflow as tf


def wine_v3_2(inputs, labels, learning_rate, mode='train', save_summary=False):
    """ The input of this model is designed to be 8 attributes
    selected from 11 original attribute,this is why i used 8
    units for dense layers. This model consisted of input, 2
    hidden layers and a final output layer
    label are numbers from 3 to 8, so the number of output units
    is 6, label will first be converted to range 0-5 and converted
    back before returning the result.
    """
    assert mode in ['train','eval'], "mode should be one of ['train', 'eval']"

    fetches = {}

    labels = labels - 3

    # stddev chosen to prevent exploding
    net = tf.layers.dense(inputs, units=8,
                          activation=tf.nn.leaky_relu,
                          name='layer1',
                          kernel_initializer=tf.initializers.random_normal(stddev=2/8))
    net = tf.layers.dense(net, units=8,
                          activation=tf.nn.leaky_relu,
                          name='layer2',
                          kernel_initializer=tf.initializers.random_normal(stddev=2/8))

    logits = tf.layers.dense(net, units=6,
                             activation=None,
                             kernel_initializer=tf.initializers.random_normal(stddev=2/8),
                             name='logits')

    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    equality = tf.equal(prediction, labels)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    fetches['accuracy'] = accuracy

    if mode == 'train':
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='softmax')
        average_loss = tf.reduce_mean(loss)
        fetches['average_loss'] = average_loss

        global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int32)
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        train_op = opt.minimize(loss, global_step=global_step)
        fetches['train_op'] = train_op
        fetches['global_step'] = global_step
    elif mode == 'eval':
        fetches['predict'] = prediction + 3

    if save_summary and mode != 'eval':
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('average_loss', average_loss)
        tf.summary.scalar('learning_rate', learning_rate)
        for tensor in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            tf.summary.histogram(tensor.name, tensor, family='trainable')
        fetches['summary_all'] = tf.summary.merge_all()

    return fetches


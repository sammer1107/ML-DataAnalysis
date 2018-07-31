import tensorflow as tf


def wine_v2(inputs, labels, learning_rate, mode='train'):
    fetches = {}
    net = tf.layers.dense(inputs, 11,
                          kernel_initializer=tf.initializers.random_normal(),
                          activation=tf.nn.tanh,
                          name='layer1')
    net = tf.layers.dense(net, 11,
                          kernel_initializer=tf.initializers.random_normal(),
                          activation=tf.nn.tanh,
                          name="layer2")
    logits = tf.layers.dense(net, 6,
                             activation=None,
                             name='logits')

    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    equality = tf.equal(prediction, labels)
    accuracy = tf.reduce_mean(tf.cast(equality, dtype=tf.float32))
    fetches['accuracy'] = accuracy

    if mode == 'train':
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='softmax_cross_entropy')
        fetches['average_loss'] = tf.reduce_mean(loss)

        global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int32)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        fetches['train_op'] = train_op
        fetches['global_step'] = global_step
    elif mode == 'eval':
        fetches['predict'] = prediction

    return fetches

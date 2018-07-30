import tensorflow as tf


def redwine_v2(inputs, labels, learning_rate, mode='train'):
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
    # one_hot_labels = tf.one_hot(labels-2, depth=max(labels-2))
    if mode == 'train':
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='softmax_cross_entropy')
        average_loss = tf.reduce_mean(loss)
        equality = tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(equality, dtype=tf.float32))
        fetches['average_loss'] = average_loss
        fetches['accuracy'] = accuracy

        global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int32)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        fetches['train_op'] = train_op
        fetches['global_step'] = global_step
    elif mode == 'eval':
        predict = tf.argmax(logits, axis=1, output_type=tf.int32)
        equality = tf.equal(predict, labels)
        accuracy = tf.reduce_mean(tf.cast(equality, dtype=tf.float32))
        fetches['accuracy'] = accuracy
        fetches['predict'] = predict
    return fetches

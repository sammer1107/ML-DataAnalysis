import tensorflow as tf
import pandas as pd
import numpy as np
import os


def predict_model(inputs):
    """
    the input should be in shape [batch_size, 7500, 4, 1] preprocessed with
    log transformation and min max scaling.
    """

    net = tf.layers.average_pooling2d(inputs, pool_size=[375, 1],
                                      strides=[375, 1],
                                      name='pooling')
    net = tf.layers.conv2d(net, filters=7,
                           kernel_size=[4,1],
                           strides=[4,1],
                           activation=tf.nn.tanh,
                           name='conv1',
                           use_bias=True)

    net = tf.reduce_mean(net, axis=[1], name='mean')
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 40, activation=tf.nn.tanh, name='dense1')
    net = tf.layers.dense(net, 6, activation=tf.nn.tanh, name='dense2')

    outputs = tf.layers.dense(net, 1, name='outputs', activation=None)
    outputs = tf.reshape(outputs,[-1])

    return tf.exp(outputs)


def main():
    files = os.listdir('test_data')
    datas = []
    for file in files:
        data = pd.read_excel('test_data/'+file, header=None)
        datas.append(np.array(data))
    datas = np.array(datas)

    # scaling pre-processing
    mins = np.array([-14.07174969, -14.12253094, -13.55425453, -14.19134808])
    maxs = np.array([-12.01072311, -11.58916759, -12.36727238, -12.87066364])
    datas = (np.log(datas)-mins)/(maxs-mins) + 0.1

    # build graph
    datas = np.expand_dims(datas, axis=3)
    datas = tf.constant(datas, dtype=tf.float32)
    outputs = predict_model(datas)

    # restore variables
    saver = tf.train.Saver()

    # restore_path = "pooled_conv2d_model_375s/checkpoints/2018-09-01-17:47-FINAL(K3)-drop3/FINAL(K3)-drop3-300000"
    # restore_path = 'pooled_conv2d_model_375s/checkpoints/2018-08-31-18:11-K3/K3-420000'
    # restore_path = 'pooled_conv2d_model_375s/checkpoints/2018-09-01-18:56-FINAL(K3)-drop4/FINAL(K3)-drop4-300000'
    restore_path = 'pooled_conv2d_model_375s/checkpoints/2018-09-01-22:58-FINAL(K3)-7/FINAL(K3)-7-170000'
    with tf.Session() as sess:
        saver.restore(sess, restore_path)
        predict = sess.run(outputs)

    with open('predict.txt', 'a') as result_file:
        result_file.write('\n'+restore_path.split('/')[-2] + '\n\n')
        for i, file in enumerate(files):
            result_file.write('{: <15} : {:.5f}\n'.format(file, predict[i]))


if __name__ == '__main__':
    main()

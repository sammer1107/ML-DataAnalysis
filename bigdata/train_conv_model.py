import tensorflow as tf
from bigdata.data.thu_dataset import ThuDataset
from bigdata.conv_model import conv_model

dataset = ThuDataset("bigdata/preprocessed_data/{}/")
inputs, targets = dataset.get_data()
inputs = tf.constant(inputs, dtype=tf.float32)
targets = tf.constant(targets, dtype=tf.float32)
fetches = conv_model(inputs, targets, 0.001)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out = {}
    for i in range(1000):
        out = sess.run(fetches)
        if (i+1) % 100 == 0:
            print('step: {},\t loss:{}'.format(out['global_step'], out['loss']))

    print(out['relative_error'])
    print(out['squared_error'])
print('Done')

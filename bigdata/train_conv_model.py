import tensorflow as tf
import datetime
from bigdata.data.thu_dataset import ThuDataset
from bigdata.conv_model import conv_model

now = datetime.datetime.now()
date = "{}-{:0>2}-{:0>2}-{:0>2}:{:0>2}".format(now.year, now.month, now.day, now.hour, now.minute)
MODEL = "conv_model"
MODE = 'train'
LR = 0.001
STEPS = 1000
RESTORE_CHK_POINT = False
RESTORE_CHK_POINT_PATH = 'bigdata/conv_model/checkpoints/2018-08-08-13:40/conv_model-1000'
SAVE_CHK_POINT = False
SAVE_SUMMARY = False

if MODE == 'eval':
    assert RESTORE_CHK_POINT, 'eval mode should be start from a trained model checkpoint'

dataset = ThuDataset("bigdata/preprocessed_data/{}/", MODE)
oinputs, otargets = dataset.get_data()
inputs = tf.constant(oinputs, dtype=tf.float32)
targets = tf.constant(otargets, dtype=tf.float32)
fetches = conv_model(inputs, targets, LR, save_summary=SAVE_SUMMARY)

if SAVE_CHK_POINT or RESTORE_CHK_POINT:
    saver = tf.train.Saver()
if SAVE_SUMMARY and MODE != 'eval':
    summary_writer = tf.summary.FileWriter('./bigdata/{}/summary/{}'.format(MODEL, date), tf.get_default_graph())

with tf.Session() as sess:

    if RESTORE_CHK_POINT:
        saver.restore(sess, RESTORE_CHK_POINT_PATH)
        print('restore variables from ', RESTORE_CHK_POINT_PATH)
    else:
        sess.run(tf.global_variables_initializer())
    if MODE == 'train':
        out = {}
        for i in range(STEPS):
            out = sess.run(fetches)
            if (i+1) % 100 == 0:
                print('step: {},\t loss:{}'.format(out['global_step'], out['loss']))

                if SAVE_SUMMARY:
                    summary_writer.add_summary(out['summary_all'], global_step=out['global_step'])
        if SAVE_CHK_POINT:
            prefix = './bigdata/{}/checkpoints/{}'.format(MODEL, date)
            saver.save(sess, prefix + '/conv_model', global_step=out['global_step'])
            if RESTORE_CHK_POINT:
                with open(prefix + '/info', "w+") as f:
                    f.write("continued from {} checkpoint".format(RESTORE_CHK_POINT_PATH))
    elif MODE == 'eval':
        out = sess.run(fetches)

    print('outputs:\n',out['outputs'])
    print('relative error:\n', out['relative_error'])
    print('squared error:\n', out['squared_error'])
    print('original input:\n', otargets)

print('Done')

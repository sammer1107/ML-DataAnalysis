import tensorflow as tf
import datetime
from bigdata.data.thu_dataset import ThuDataset
from bigdata.conv_model import conv_model

now = datetime.datetime.now()
date = "{}-{:0>2}-{:0>2}-{:0>2}:{:0>2}".format(now.year, now.month, now.day, now.hour, now.minute)
MODEL = "conv_model"
MODE = 'train'
LR = 0.001
STEPS = 30
RESTORE_CHK_POINT = False
RESTORE_CHK_POINT_PATH = ''
SAVE_CHK_POINT = False
SAVE_SUMMARY = False


dataset = ThuDataset("bigdata/preprocessed_data/{}/", MODE)
inputs, targets = dataset.get_data()
inputs = tf.constant(inputs, dtype=tf.float32)
targets = tf.constant(targets, dtype=tf.float32)
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
            print(sess.run(['conv1/kernel:0']))
            out = sess.run(fetches)
            if (i+1) % 1 == 0:
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

    print('relative error:\n', out['relative_error'])
    print('squared error:\n', out['squared_error'])

print('Done')

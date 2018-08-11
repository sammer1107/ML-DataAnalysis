import tensorflow as tf
import datetime
import numpy as np
from bigdata.data.thu_dataset import ThuDataset
from bigdata.conv_model import pooled_conv_model

now = datetime.datetime.now()
date = "{}-{:0>2}-{:0>2}-{:0>2}:{:0>2}".format(now.year, now.month, now.day, now.hour, now.minute)
MODEL = "pooled_conv_model"
MODE = 'train'
LR = 20.0
LR_DECAY = 0.96
STEPS = 100000
RESTORE_CHK_POINT = True
RESTORE_CHK_POINT_PATH = 'bigdata/pooled_conv_model/checkpoints/2018-08-09-16:38/conv_model-1300000'
SAVE_CHK_POINT = True
SAVE_CHK_POINT_STEP = 20000
SAVE_SUMMARY = True
CHECKPOINT_PATH = './bigdata/{}/checkpoints/{}'.format(MODEL, date)

if MODE == 'eval':
    assert RESTORE_CHK_POINT, 'eval mode should be start from a trained model checkpoint'


def main():

    dataset = ThuDataset("bigdata/preprocessed_data/{}/", MODE)
    oinputs, otargets = dataset.get_data()
    inputs = tf.constant(oinputs, dtype=tf.float32)
    targets = tf.constant(otargets, dtype=tf.float32)
    fetches = pooled_conv_model(inputs, targets, LR, learning_rate_decay=LR_DECAY, save_summary=SAVE_SUMMARY)

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
                if SAVE_CHK_POINT and (i+1) % SAVE_CHK_POINT_STEP == 0 :
                    saver.save(sess, CHECKPOINT_PATH + '/conv_model', global_step=out['global_step'])

            if SAVE_CHK_POINT and RESTORE_CHK_POINT:
                with open(CHECKPOINT_PATH + '/info', "w+") as f:
                    f.write("continued from {} checkpoint".format(RESTORE_CHK_POINT_PATH))
        elif MODE == 'eval':
            out = sess.run(fetches)

        print('original input:\n', pretty_print(otargets))
        print('outputs:\n', pretty_print(out['outputs']))
        print('relative error:\n', pretty_print(out['relative_error']))
        print('squared error:\n', pretty_print(out['squared_error']))

    print('Done')


def pretty_print(x):
    repr_str = np.array_repr(x)
    repr_str = repr_str.replace(' ','').replace('\n','')[7:-16]
    splits = np.array(repr_str.split(',')).astype(np.float32)
    template = '{:+.5f},\t'*(len(splits)-1) + '{:+5f}'
    formatted = template.format(*splits)
    return formatted


main()

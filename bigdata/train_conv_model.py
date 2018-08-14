import tensorflow as tf
import datetime
import numpy as np
from bigdata.data.thu_dataset import ThuDataset
from bigdata.conv_model import *

now = datetime.datetime.now()
date = "{}-{:0>2}-{:0>2}-{:0>2}:{:0>2}".format(now.year, now.month, now.day, now.hour, now.minute)
MODEL = "pooled_conv_model"
note = 'tanh8+1'
MODE = 'eval'
LR = 0.005
LR_DECAY = 0.99
STEPS = 200000
RESTORE_CHK_POINT = True
RESTORE_CHK_POINT_PATH = 'bigdata/pooled_conv_model/checkpoints/2018-08-14-16:18-tanh8+1/conv_model-500000'
SAVE_CHK_POINT = True
SAVE_CHK_POINT_STEP = 50000
SAVE_SUMMARY = True

if RESTORE_CHK_POINT:
    CHECKPOINT_PATH = RESTORE_CHK_POINT_PATH.rsplit('/',1)[0]
    SUMMARY_PATH = './bigdata/{}/summary/{}'.format(MODEL, RESTORE_CHK_POINT_PATH.rsplit('/')[-2])
else:
    CHECKPOINT_PATH = './bigdata/{}/checkpoints/{}-{}'.format(MODEL, date, note)
    SUMMARY_PATH = './bigdata/{}/summary/{}-{}'.format(MODEL, date, note)

if MODE == 'eval':
    assert RESTORE_CHK_POINT, 'eval mode should be start from a trained model checkpoint'


def main():

    dataset = ThuDataset("bigdata/log_normalized_data/{}/", MODE)
    oinputs, otargets = dataset.get_data()
    inputs = tf.constant(oinputs, dtype=tf.float32)
    targets = tf.constant(otargets, dtype=tf.float32)
    fetches = pooled_conv_model(inputs, targets, LR, learning_rate_decay=LR_DECAY, save_summary=SAVE_SUMMARY)

    if SAVE_CHK_POINT or RESTORE_CHK_POINT:
        saver = tf.train.Saver(max_to_keep=10)
    if SAVE_SUMMARY and MODE != 'eval':
        summary_writer = tf.summary.FileWriter(SUMMARY_PATH, tf.get_default_graph())

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
                    print('step: {: >7},\t loss:{:.5E}'.format(out['global_step'], out['loss']))
                    if SAVE_SUMMARY:
                        summary_writer.add_summary(out['summary_all'], global_step=out['global_step'])
                if SAVE_CHK_POINT and (i+1) % SAVE_CHK_POINT_STEP == 0:
                    saver.save(sess, CHECKPOINT_PATH + '/conv_model', global_step=out['global_step'])
            # no longer needed
            # if SAVE_CHK_POINT and RESTORE_CHK_POINT:
            #     with open(CHECKPOINT_PATH + '/info', "w+") as f:
            #         f.write("continued from {} checkpoint".format(RESTORE_CHK_POINT_PATH))
        elif MODE == 'eval':
            out = sess.run(fetches)

        print('original input:\n', pretty_print(otargets))
        print('outputs:\n', pretty_print(out['outputs']))
        print('error:\n', pretty_print(out['outputs']-otargets))
        exp_error = np.exp(otargets)-np.exp(out['outputs'])
        print('exp error:\n', pretty_print(exp_error))
        print('average exp error:\n', np.mean(np.abs(exp_error)))
        print('RMSE: ', np.sqrt(out['loss']))
        print('exp RMSE:\n', np.sqrt(np.mean(np.square(exp_error))))
        # print('original input:\n', pretty_print(np.exp(otargets)))
        # print('outputs:\n', pretty_print(np.exp(out['outputs'])))

    print('Done')


def pretty_print(x):
    repr_str = np.array_repr(x)
    repr_str = repr_str.replace(' ','').replace('\n','')[7:-16]
    splits = np.array(repr_str.split(',')).astype(np.float32)
    template = '{:+.5f},\t'*(len(splits)-1) + '{:+5f}'
    formatted = template.format(*splits)
    return formatted


if __name__ == '__main__':
    main()

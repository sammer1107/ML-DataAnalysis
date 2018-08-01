import tensorflow as tf
import numpy as np
import datetime
from wine.data.redwine_data import RedWine
from wine.wine_model_v3 import wine_v3_2

now = datetime.datetime.now()
date = "{}-{:0>2}-{:0>2}-{:0>2}:{:0>2}".format(now.year, now.month, now.day, now.hour, now.minute)
MODEL = "wine_model_v3_2"
MODE = 'train'
LR = 0.0001
STEPS = 100000
RESTORE_CHK_POINT = True
RESTORE_CHK_POINT_PATH = './wine/wine_model_v3_2/checkpoints/2018-08-01-20:56/redwine-300000'
SAVE_CHK_POINT = True
SAVE_SUMMARY = True

redwine_data = RedWine(subset=MODE)
input_data, label_data = redwine_data.get_data()
inputs = tf.constant(input_data, dtype=tf.float32)
labels = tf.constant(label_data, dtype=tf.int32)
fetches = wine_v3_2(inputs, labels, LR, mode=MODE, save_summary=SAVE_SUMMARY, save_checkpoint=True)

if SAVE_CHK_POINT or RESTORE_CHK_POINT:
    saver = tf.train.Saver(max_to_keep=10)
if SAVE_SUMMARY and MODE != 'eval':
    summary_writer = tf.summary.FileWriter('./wine/{}/summary/{}'.format(MODEL, date), tf.get_default_graph())

with tf.Session() as sess:

    if RESTORE_CHK_POINT:
        saver.restore(sess, RESTORE_CHK_POINT_PATH)
        print('restore variables from ', RESTORE_CHK_POINT_PATH)
    else:
        sess.run(tf.global_variables_initializer())

    if MODE == 'train':
        for step in range(STEPS):
            out = sess.run(fetches)
            if (step + 1) % 1000 == 0:
                print('train_step: {}\t loss: {:.4f}\taccuracy: {:.4f}'.format(out['global_step'],
                                                                               out['average_loss'],
                                                                               out['accuracy']))
                if SAVE_SUMMARY:
                    summary_writer.add_summary(out['summary_all'], global_step=out['global_step'])

        if SAVE_CHK_POINT:
            prefix = './wine/{}/checkpoints/{}'.format(MODEL, date)
            saver.save(sess, prefix + '/redwine', global_step=out['global_step'])
            if RESTORE_CHK_POINT:
                with open(prefix + '/info', "w+") as f:
                    f.write("continued from {} checkpoint".format(RESTORE_CHK_POINT_PATH))

    elif MODE == 'eval':
        out = sess.run(fetches)
        predict_out = np.array(out['predict'], dtype=np.float32)
        for i, prediction in enumerate(predict_out):
            print('{} answer: {}\tprediction: {}'.format(i, label_data[i], prediction))
        print(out['accuracy'])


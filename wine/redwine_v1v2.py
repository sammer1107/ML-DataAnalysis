import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
from wine.wine_model_v2 import wine_v2

now = datetime.datetime.now()
date = "{}-{:0>2}-{:0>2}-{:0>2}:{:0>2}".format(now.year, now.month, now.day, now.hour, now.minute)
model_name = "wine_model_v2"
MODE = 'eval'
LR = 0.00006
STEP = 500000
RESTORE_CHK_POINT = True
RESTORE_CHK_POINT_PATH = './wine_model_v2/checkpoints/2018-07-29-18:41/redwine-1500000'
SAVE_CHK_POINT = True
SAVE_SUMMARY = True

redwine_data = pd.read_csv("winequality-red.csv")
# print(np.shape(np.array(redwine_data)))
keys = redwine_data.keys()
num_keys = len(redwine_data.keys())
redwine_data = np.array(redwine_data, dtype=np.float32)
redwine_data[:,:-1] /= np.max(redwine_data[:,:-1], axis=0)

# print('keys', keys)
# plt.figure()

# for i in range(num_keys-1):
#     plt.subplot(num_keys // 4 + 1, 4, i+1)
#     plt.scatter(redwine_data[keys[i]] / np.max(redwine_data[keys[i]]), redwine_data['quality'])
#     plt.xlabel(keys[i])
#
# plt.tight_layout()
# plt.show()

if MODE == 'train':
    # get rid of label, make data in range 0-1
    inputs = redwine_data[:,:-1]
    # print(inputs[0:3,:])
    inputs = tf.constant(inputs, dtype=tf.float32)
    labels = tf.constant(redwine_data[:,-1] - 3, dtype=tf.int32)
    # print(np.unique(labels))
elif MODE == 'eval':
    selection = redwine_data[np.random.choice(len(redwine_data), [100])]
    inputs = selection[:,:-1]
    eval_labels = selection[:, -1] - 3
    inputs = tf.constant(inputs, dtype=tf.float32)
    labels = tf.constant(eval_labels, dtype=tf.int32)


fetches = wine_v2(inputs, labels, learning_rate=LR, mode=MODE)

# shape_output = tf.shape(output)
# shape_label = tf.shape(one_hot_labels)
if SAVE_CHK_POINT:
    saver = tf.train.Saver(max_to_keep=10)
if RESTORE_CHK_POINT:
    saver = tf.train.Saver(max_to_keep=10)
    # inspect_chkp.print_tensors_in_checkpoint_file(RESTORE_CHK_POINT_PATH, tensor_name='', all_tensors=True)
if SAVE_SUMMARY and MODE == 'train':
    tf.summary.scalar("average_loss", fetches['average_loss'])
    tf.summary.scalar("accuracy", fetches['accuracy'])
    tf.summary.scalar("learning_rate", LR)
    for tensor in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print(tensor)
        tf.summary.histogram(tensor.name, tensor)
    summary_all = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./{}/summary/{}'.format(model_name, date), tf.get_default_graph())
    fetches['summary_all'] = summary_all

########### Session ###########

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    if RESTORE_CHK_POINT:
        saver.restore(sess, RESTORE_CHK_POINT_PATH)
        print('restore variables from ', RESTORE_CHK_POINT_PATH)
    else:
        sess.run(tf.global_variables_initializer())

    if MODE == 'train':
        for step in range(STEP):
            out = sess.run(fetches)
            if step+1 % 1000 == 0:
                print('train_step: {}\t loss: {:.4f}\taccuracy: {:.4f}'.format(out['global_step'],
                                                                               out['average_loss'],
                                                                               out['accuracy']))
                if SAVE_SUMMARY:
                    summary_writer.add_summary(out['summary_all'], global_step=out['global_step'])

        if SAVE_CHK_POINT:
            prefix = './{}/checkpoints/{}'.format(model_name, date)
            saver.save(sess, prefix + '/redwine', global_step=out['global_step'])
            if RESTORE_CHK_POINT:
                with open(prefix + '/info', "w+") as f:
                    f.write("continued from {} checkpoint".format(RESTORE_CHK_POINT_PATH))

    elif MODE == 'eval':
        out = sess.run(fetches)
        predict_out = np.array(out['predict'], dtype=np.float32)
        for i,prediction in enumerate(predict_out):
            print('{} answer: {}\tprediction: {}'.format(i, eval_labels[i]+3, prediction+3))
        print(out['accuracy'])

import tensorflow as tf
import copy
import numpy as np
from bigdata.data.thu_dataset import ThuDataset
from bigdata.utils import *
from bigdata.constants import *
from bigdata import conv_model


# ===================================== SETTINGS ======================================= #
date = format_date()
MODEL = "pooled_conv2d_model_375sd"
note = '5(soft+)-mean-1(lin+soft)LR4e-4'
MODE = EVAL
LR = 0.0004
LR_DECAY = 1
BATCH_SIZE = 35
STEPS = 100000
RESTORE_CHK_POINT = True
KEEP_RESTORE_DIR = True
RESTORE_PART = False
RESTORE_LIST = {'conv1/kernel':'conv1/kernel:0','conv1/bias':'conv1/bias:0',
                'dense/linear/kernel':'dense/linear/kernel:0',
                'dense/post_bias':'dense/post_bias:0',
                'dense/pre_bias':'dense/pre_bias:0',
                'dense/pre_scale':'dense/pre_scale:0',
                'dense/post_scale':'dense/post_scale:0'}
EVAL_RANGE = (1520000,2000001,20000)
RESTORE_CHK_POINT_PATH = \
    'bigdata/pooled_conv2d_model_375sd/checkpoints/2018-08-22-22:39-5(soft+)-mean-1(lin+soft)/5(soft+)-mean-1(lin+soft)LR4e-4-{}'
SAVE_CHK_POINT = True
SAVE_CHK_POINT_STEP = 20000
SUMMARY_LV = 2
SUMMARY_STEP = 1000
SAVE_STRATEGY = BY_STEP
# ===================================================================================== #

# === decide if should create new folder for this run === #
if RESTORE_CHK_POINT and KEEP_RESTORE_DIR:
    CHECKPOINT_PATH = RESTORE_CHK_POINT_PATH.rsplit('/',1)[0]
    SUMMARY_PATH = './bigdata/{}/summary/{}'.format(MODEL, RESTORE_CHK_POINT_PATH.rsplit('/')[-2])
else:
    CHECKPOINT_PATH = './bigdata/{}/checkpoints/{}-{}'.format(MODEL, date, note)
    SUMMARY_PATH = './bigdata/{}/summary/{}-{}'.format(MODEL, date, note)
# ======================================================= #

if MODE == EVAL:
    assert RESTORE_CHK_POINT, 'eval mode should be start from a trained model checkpoint'


def main():

    dataset = ThuDataset("bigdata/log_minmax_data/{}/", MODE)
    oinputs, otargets = dataset.get_data()
    # reduce columns
    # reduced_inputs = np.zeros([np.shape(oinputs)[0],7500,2,1])
    # reduced_inputs[:,:,1] = oinputs[:,:,2]
    # reduced_inputs[:,:,0] = (oinputs[:,:,0] + oinputs[:,:,1] + oinputs[:,:,3])/3
    # transform to tensors
    inputs = tf.constant(oinputs, dtype=tf.float32)
    targets = tf.constant(otargets, dtype=tf.float32)
    # generate graph
    model = getattr(conv_model, MODEL)
    fetches = model(inputs, targets, LR, batch_size=BATCH_SIZE, learning_rate_decay=LR_DECAY, mode=MODE, summary_lv=SUMMARY_LV)
    graph = tf.get_default_graph()

    if SAVE_CHK_POINT or RESTORE_CHK_POINT:
        max_to_keep = 10 if SAVE_STRATEGY == BY_LOSS else 100
        saver = tf.train.Saver(max_to_keep=max_to_keep)
    if RESTORE_PART:
        assert RESTORE_LIST, 'No RESTORE_LIST specified'

        for key, value in RESTORE_LIST.items():
            RESTORE_LIST[key] = graph.get_tensor_by_name(value)
        restore_saver = tf.train.Saver(var_list=RESTORE_LIST)
    if SUMMARY_LV and MODE != EVAL:
        summary_writer = tf.summary.FileWriter(SUMMARY_PATH, tf.get_default_graph())

    config = tf.ConfigProto()
    # config.log_device_placement = True

    with tf.Session(config=config) as sess:
        if RESTORE_CHK_POINT:
            if not RESTORE_PART:
                saver.restore(sess, RESTORE_CHK_POINT_PATH)
                print('restore variables from ', RESTORE_CHK_POINT_PATH)
            else:
                sess.run(tf.global_variables_initializer())
                restore_saver.restore(sess, RESTORE_CHK_POINT_PATH)
                print('restore variables from ', RESTORE_CHK_POINT_PATH)
                print(RESTORE_LIST)
        else:
            sess.run(tf.global_variables_initializer())
        if MODE == TRAIN:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            out = {}
            log_loss_count = np.log2(sess.run(fetches['loss']))
            for i in range(STEPS):

                out = sess.run(fetches)

                if (i+1) % 100 == 0:
                    print('step: {: >7},\t loss: {:.5E}\t({:.5E})'.format(
                        out['global_step'], out['regularized_loss'], out['loss']))
                if (i+1) % SUMMARY_STEP == 0:
                    if SUMMARY_LV:
                        summary_writer.add_summary(out['summary_all'], global_step=out['global_step'])

                if SAVE_STRATEGY == BY_STEP:
                    if SAVE_CHK_POINT and (i+1) % SAVE_CHK_POINT_STEP == 0:
                        saver.save(sess, '{}/{}'.format(CHECKPOINT_PATH, note), global_step=out['global_step'])
                elif SAVE_STRATEGY == BY_LOSS:
                    log_loss = np.ceil(np.log2(out['loss']))
                    if log_loss < log_loss_count:
                        log_loss_count = log_loss
                        saver.save(sess, CHECKPOINT_PATH + '/conv_model', global_step=out['global_step'])
                        print('saved checkpoint of log10(loss) = {}'.format(log_loss_count))

            if SAVE_STRATEGY == BY_LOSS:  # final save
                saver.save(sess, CHECKPOINT_PATH + '/conv_model', global_step=out['global_step'])
            coord.request_stop()
            coord.join(threads)
        elif MODE == EVAL:
            out = sess.run(fetches)

        exp_error = np.exp(out['targets'])-np.exp(out['outputs'])
        average_exp_error = np.mean(np.abs(exp_error))
        exp_rmse = np.sqrt(np.mean(np.square(exp_error)))
        print('original input:\n', pretty_print(out['targets']))
        print('outputs:\n', pretty_print(out['outputs']))
        print('error:\n', pretty_print(out['outputs']-out['targets']))
        print('exp original inputs:\n', pretty_print(np.exp(out['targets'])))
        print('exp error:\n', pretty_print(exp_error))
        print('average exp error:\n{:.10f}'.format(average_exp_error))
        print('RMSE:\n{:.10f}'.format(np.sqrt(out['loss'])))
        print('exp RMSE:\n{:.10f}'.format(exp_rmse))

        if MODE == EVAL:
            path, file = RESTORE_CHK_POINT_PATH.rsplit('/', 1)
            with open(path + '/eval', 'a') as eval_result:
                eval_result.write('{}\n{:.10f} ({:.10f})\n'.format(file, exp_rmse, average_exp_error))

    print('Done')


if __name__ == '__main__':
    if MODE == TRAIN:
        main()
    elif MODE == EVAL:
        O_RESTORE_CHK_POINT_PATH = RESTORE_CHK_POINT_PATH
        for i in range(*EVAL_RANGE):
            RESTORE_CHK_POINT_PATH = O_RESTORE_CHK_POINT_PATH.format(i)
            main()
            tf.reset_default_graph()

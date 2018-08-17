import tensorflow as tf
import datetime
import numpy as np
from bigdata.data.thu_dataset import ThuDataset
from bigdata import conv_model

# constants
EVAL = 'eval'
TRAIN = 'train'
BY_STEP = 'by_step'
BY_LOSS = 'by_loss'

now = datetime.datetime.now()
date = "{}-{:0>2}-{:0>2}-{:0>2}:{:0>2}".format(now.year, now.month, now.day, now.hour, now.minute)
MODEL = "pooled_conv2d_model_250dd"
note = '222f-batch1'
MODE = EVAL
LR = 0.0001
LR_DECAY = 0.99
BATCH_SIZE = 4
STEPS = 100000
RESTORE_CHK_POINT = True
RESTORE_CHK_POINT_PATH = \
    'bigdata/pooled_conv2d_model_250dd/checkpoints/2018-08-17-22:25-222f-batch1/conv_model-350000'
SAVE_CHK_POINT = True
SAVE_CHK_POINT_STEP = 20000
SUMMARY_LV = 2
SUMMARY_STEP = 1000
SAVE_STRATEGY = BY_LOSS

if RESTORE_CHK_POINT:
    CHECKPOINT_PATH = RESTORE_CHK_POINT_PATH.rsplit('/',1)[0]
    SUMMARY_PATH = './bigdata/{}/summary/{}'.format(MODEL, RESTORE_CHK_POINT_PATH.rsplit('/')[-2])
else:
    CHECKPOINT_PATH = './bigdata/{}/checkpoints/{}-{}'.format(MODEL, date, note)
    SUMMARY_PATH = './bigdata/{}/summary/{}-{}'.format(MODEL, date, note)

if MODE == EVAL:
    assert RESTORE_CHK_POINT, 'eval mode should be start from a trained model checkpoint'


def main():

    dataset = ThuDataset("bigdata/log_normalized_data/{}/", MODE)
    oinputs, otargets = dataset.get_data()
    inputs = tf.constant(oinputs, dtype=tf.float32)
    targets = tf.constant(otargets, dtype=tf.float32)
    model = getattr(conv_model, MODEL)
    fetches = model(inputs, targets, LR, BATCH_SIZE, LR_DECAY, mode=MODE, summary_lv=SUMMARY_LV)

    if SAVE_CHK_POINT or RESTORE_CHK_POINT:
        max_to_keep = 5 if SAVE_STRATEGY == BY_LOSS else 10
        saver = tf.train.Saver(max_to_keep=max_to_keep)
    if SUMMARY_LV and MODE != EVAL:
        summary_writer = tf.summary.FileWriter(SUMMARY_PATH, tf.get_default_graph())

    with tf.Session() as sess:
        if RESTORE_CHK_POINT:
            saver.restore(sess, RESTORE_CHK_POINT_PATH)
            print('restore variables from ', RESTORE_CHK_POINT_PATH)
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
                    print('step: {: >7},\t loss: {:.5E}'.format(out['global_step'], out['loss']))
                if (i+1) % SUMMARY_STEP == 0:
                    if SUMMARY_LV:
                        summary_writer.add_summary(out['summary_all'], global_step=out['global_step'])

                if SAVE_STRATEGY == BY_STEP:
                    if SAVE_CHK_POINT and (i+1) % SAVE_CHK_POINT_STEP == 0:
                        saver.save(sess, CHECKPOINT_PATH + '/conv_model', global_step=out['global_step'])
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

        print('original input:\n', pretty_print(out['inputs']))
        print('outputs:\n', pretty_print(out['outputs']))
        print('error:\n', pretty_print(out['outputs']-out['inputs']))
        exp_error = np.exp(out['inputs'])-np.exp(out['outputs'])
        print('exp original inputs:\n', pretty_print(np.exp(out['inputs'])))
        print('exp error:\n', pretty_print(exp_error))
        print('average exp error:\n{:.10f}'.format(np.mean(np.abs(exp_error))))
        print('RMSE:\n{:.10f}'.format(np.sqrt(out['loss'])))
        print('exp RMSE:\n{:.10f}'.format(np.sqrt(np.mean(np.square(exp_error)))))
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

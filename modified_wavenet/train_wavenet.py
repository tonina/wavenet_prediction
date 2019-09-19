import os
import sys
from datetime import datetime
import threading

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from modified_wavenet.wavenet_model import *
from process_data import DataTrainBatch
from hparams import Parameters as p

tf.set_random_seed(1)
np.random.seed(1)


def wavenet_train(model_path, model_name, num_steps, start_step=0, tune=False):

    coord = tf.train.Coordinator()
    b_size = 1

    x = tf.placeholder(tf.float32, shape=(b_size, None))
    c = tf.placeholder(tf.float32, shape=(b_size, None, p.num_mels))

    queue = tf.RandomShuffleQueue(capacity=10,
                                  min_after_dequeue=5,
                                  dtypes=[tf.float32, tf.float32])
    enqueue_op = queue.enqueue([x, c])
    ins, mels = queue.dequeue()
    ins.set_shape(x.shape)
    mels.set_shape(c.shape)

    print('ins', ins)
    graph = WavenetGraph(ins, mels)
    graph.add_loss()
    graph.add_optimizer()
    print('Create graph')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if tune:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        else:
            sess.run(tf.global_variables_initializer())

        def enqueue_thread():
            with coord.stop_on_exception():
                while not coord.should_stop():
                    w_batch = DataTrainBatch(b_size)
                    while True:
                        try:
                            xb, cb = next(w_batch)
                            # x_quantized = mu_law_encode(xb, p.quantization_channels)
                            # x_hot = tf.one_hot(x_quantized, p.quantization_channels)
                            if p.repeat_conditions:
                                cb = np.repeat(cb, 256, axis=1)
                            sess.run(enqueue_op, feed_dict={x: xb, c: cb})
                        except StopIteration:
                            break

        threading.Thread(target=enqueue_thread, args=()).start()

        for step in range(start_step, start_step+num_steps, 1):
            print(step)

            loss, tr_op = sess.run([graph.loss, graph.train_op])
            print('step - {}, loss = {}'.format(step, loss))

            if step % 100 == 0:
                saver.save(sess, os.path.join(model_path, model_name))
        saver.save(sess, os.path.join(model_path, model_name))


if __name__ == '__main__':
    wavenet_train('..\\w_models\\model1', 'w_model1', 10)

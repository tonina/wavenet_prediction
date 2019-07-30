import os

import numpy as np
import tensorflow as tf

from modified_wavenet.wavenet_modules import *
# from utils.audio_utils import *
# from utils.repeats import tf_repeat

from hparams import Parameters as p

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.set_random_seed(1)
np.random.seed(1)


def get_receptive_field(num_layers, num_cycles, kernel_size, dilation=lambda x: 2**x):

    layers_per_cycle = num_layers // num_cycles
    dilations = [dilation(i % layers_per_cycle) for i in range(num_layers)]
    receptive_field = (kernel_size - 1) * sum(dilations) + 1

    return receptive_field


class WavenetGraph(object):
    def __init__(self, inputs=None, cond_inputs=None, mode='train'):
        self.num_layers = p.wavenet_num_layers
        self.dilation_channels = p.dilation_channels
        self.residual_channels = p.residual_channels
        self.skip_channels = p.skip_channels
        self.quantization_channels = p.quantization_channels
        self.filter_width = p.filter_width

        if mode == 'train':
            self.inputs = inputs

        self.quantized_inputs = mu_law_encode(self.inputs, p.quantization_channels)
        self.hot_inputs = tf.one_hot(self.quantized_inputs, p.quantization_channels)

        # self.quantized = mu_law_encode(self.inputs, self.quantization_channels)
        # self.hot_encoded = tf.one_hot(self.inputs, depth=self.quantization_channels)
        first_conv_layer = CausalConv1D(self.dilation_channels, self.filter_width, activation=tf.nn.tanh)
        self.first_conv = first_conv_layer(self.hot_inputs)

        x = self.first_conv
        skips = self.first_conv
        local_inputs = None

        # local conditional inputs upsampling
        if p.local_condition and p.repeat_conditions and (cond_inputs is not None):
            local_inputs = cond_inputs
            # local_inputs = tf_repeat(cond_inputs, [1, 256, 1])
            # local_inputs = np.repeat(cond_inputs, 256, axis=1)

        for k in range(p.wavenet_num_layers):
            dilation = 2**(k % 10)
            gated_unit = GatedUnit(dilation_rate=dilation,
                                   filters=self.dilation_channels,
                                   residual_channels=self.residual_channels,
                                   skip_channels=self.skip_channels)
            skip, x = gated_unit(x, local_inputs)
            skips = tf.concat([skips, skip], axis=-1)
        self.after_dilated = skips
        self.relu_dilated = tf.nn.relu(self.after_dilated)
        self.prelast_conv = tf.layers.conv1d(self.relu_dilated, self.quantization_channels, self.filter_width,
                                             padding='same', activation=tf.nn.relu)
        self.outputs = tf.layers.conv1d(self.prelast_conv, self.quantization_channels, self.filter_width,
                                          padding='same')
        # self.outputs = tf.argmax(self.last_conv, axis=-1)
        # self.outputs = mu_law_decode(self.hot_decoded, self.quantization_channels)

    def add_loss(self):
        with tf.variable_scope('loss') as scope:
            self.loss = tf.losses.mean_squared_error(self.outputs, self.hot_inputs)

    def add_optimizer(self):
        with tf.variable_scope('optimizer') as scope:
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
            self.train_op = optimizer.minimize(self.loss)


# if __name__ == '__main__':
#     w_batch = WavBatch(1)
#     mels, wavs = next(w_batch)
#
#     quantized = mu_law_encode(wavs, p.quantization_channels)
#     ins = tf.one_hot(quantized, p.quantization_channels)
#
#     if p.repeat_conditions:
#         conds = np.repeat(mels, 256, axis=1)
#     else:
#         conds = mels
#
#     print('ins shape', ins.shape)
#     print('mels shape', conds.shape)
#
#     graph = WavenetGraph(ins, conds)
#     graph.add_loss()
#     print(tf.trainable_variables())
#     graph.add_optimizer()
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         print(tf.trainable_variables())
#         outs = sess.run([graph.first_conv,
#                          graph.after_dilated,
#                          graph.prelast_conv,
#                          graph.outputs])
#         for item in outs:
#             print(item.shape)
#         for step in range(100):
#             tr_op, loss = sess.run([graph.train_op, graph.loss])
#             print('step - {}, loss ={}'.format(step, loss))

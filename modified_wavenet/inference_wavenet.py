import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf

from wavenet.wavenet_model import WavenetGraph, get_receptive_field
from utils.data_utils import get_spectrogram
from hparams import Parameters as p


tf.reset_default_graph()


wav_file = 'e:\\Data\\text_to_speech\\LJSpeech-1.1-onetest\\wavs\\LJ001-0001.wav'
mel, _, _ = get_spectrogram(wav_file)
ex_mel = np.repeat(mel, 256, axis=0)
# rec_field = get_receptive_field(30, 3, 2)
rec_field = 4000
start_mel = np.zeros((rec_field, ex_mel.shape[-1]), dtype=np.float32)
cond_ins = np.concatenate((start_mel, ex_mel), axis=0).astype(np.float32)
cond_in = cond_ins[0: 0+rec_field]
start_sig = np.zeros((rec_field,), dtype=np.float32)
seq_in = start_sig
cond_in = np.expand_dims(cond_in, axis=0)
seq_in = np.expand_dims(seq_in, axis=0)
print('cond_in shape', cond_in.shape)
print('seq_in shape', seq_in.shape)
g = WavenetGraph(inputs=seq_in, cond_inputs=cond_in)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('w_models\\model1'))
    # for i in range(ex_mel.shape[0]):
    for i in range(1):
        cond_in = cond_ins[i: i+rec_field]
        seq_in = start_sig
        g = WavenetGraph(inputs=seq_in, cond_inputs=cond_in)
        outs = sess.run(g.outputs)
        print(outs)

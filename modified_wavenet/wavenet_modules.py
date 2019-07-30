import tensorflow as tf


class CausalConv1D(tf.layers.Conv1D):
    def __init__(self, filters,
               kernel_size,
               strides=1,
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
        )

    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
        return super(CausalConv1D, self).call(inputs)


class GatedUnit(object):
    def __init__(self, dilation_rate, filters, residual_channels, skip_channels):
        self.dilation_rate = dilation_rate
        self.filters = filters
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels

    def __call__(self, inputs, cond_inputs=None):
        dilated_conv = CausalConv1D(filters=self.filters,
                                    kernel_size=2,
                                    dilation_rate=self.dilation_rate)
        conv = dilated_conv(inputs)
        if cond_inputs is not None:
            conv = tf.concat([conv, cond_inputs], axis=-1)
        tanh_block = tf.nn.tanh(conv)
        sigmoid_block = tf.nn.sigmoid(conv)
        multiplied = tf.multiply(tanh_block, sigmoid_block)

        residual_conv = tf.layers.conv1d(multiplied,
                                         filters=self.residual_channels,
                                         kernel_size=1)
        skips = tf.layers.conv1d(multiplied,
                                     filters=self.skip_channels,
                                     kernel_size=1)
        residuals = tf.concat([inputs, residual_conv], axis=-1)

        return skips, residuals


def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude

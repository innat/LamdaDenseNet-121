immport tensorflow as tf 

class MixDepthGroupConvolution2D(tf.keras.layers.Layer):
    def __init__(self, kernels=[3, 5, 7],
                 conv_kwargs=None,
                 **kwargs):
        super(MixDepthGroupConvolution2D, self).__init__(**kwargs)

        if conv_kwargs is None:
            conv_kwargs = {
                'strides': (1, 1),
                'padding': 'same',
                'dilation_rate': (1, 1),
                'use_bias': False,
            }
        self.channel_axis = -1 
        self.kernels = kernels
        self.groups = len(self.kernels)
        self.strides = conv_kwargs.get('strides', (1, 1))
        self.padding = conv_kwargs.get('padding', 'same')
        self.dilation_rate = conv_kwargs.get('dilation_rate', (1, 1))
        self.use_bias = conv_kwargs.get('use_bias', False)
        self.conv_kwargs = conv_kwargs or {}

        self.layers = [tf.keras.layers.DepthwiseConv2D(kernels[i],
                                       strides=self.strides,
                                       padding=self.padding,
                                       activation=tf.nn.relu,                
                                       dilation_rate=self.dilation_rate,
                                       use_bias=self.use_bias,
                                       kernel_initializer='he_normal')
                        for i in range(self.groups)]

    def call(self, inputs, **kwargs):
        if len(self.layers) == 1:
            return self.layers[0](inputs)
        filters = K.int_shape(inputs)[self.channel_axis]
        splits  = self.split_channels(filters, self.groups)
        x_splits  = tf.split(inputs, splits, self.channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self.layers)]
        return tf.keras.layers.concatenate(x_outputs, 
                                           axis=self.channel_axis)

    def split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def get_config(self):
        config = {
            'kernels': self.kernels,
            'groups': self.groups,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'conv_kwargs': self.conv_kwargs,
        }
        base_config = super(MixDepthGroupConvolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Parameters(object):

    # wavenet parameters
    filter_width = 2
    dilation_channels = 32
    residual_channels = 32
    skip_channels = 64
    quantization_channels = 256
    initial_filter_width = 32
    wavenet_num_layers = 30
    local_condition = True
    repeat_conditions = True

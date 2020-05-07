import tensorflow as tf
import math

from configuration import Config


def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, ratio=0.25):
        super(SEBlock, self).__init__()
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reduce_conv = tf.keras.layers.Conv2D(filters=self.num_reduced_filters,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")
        self.expand_conv = tf.keras.layers.Conv2D(filters=input_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = self.reduce_conv(branch)
        branch = tf.nn.swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.nn.sigmoid(branch)
        output = inputs * branch
        return output


class MBConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.conv1 = tf.keras.layers.Conv2D(filters=in_channels * expansion_factor,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                      strides=stride,
                                                      padding="same",
                                                      use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se = SEBlock(input_channels=in_channels * expansion_factor)
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=drop_connect_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = self.se(x)
        x = tf.nn.swish(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x = self.dropout(x, training=training)
            x = tf.keras.layers.add([x, inputs])
        return x


def build_mbconv_block(in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate):
    block = tf.keras.Sequential()
    for i in range(layers):
        if i == 0:
            block.add(MBConv(in_channels=in_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=stride,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
        else:
            block.add(MBConv(in_channels=out_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=1,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
    return block


class EfficientNet(tf.keras.Model):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate=0.2):
        super(EfficientNet, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=round_filters(32, width_coefficient),
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.block1 = build_mbconv_block(in_channels=round_filters(32, width_coefficient),
                                         out_channels=round_filters(16, width_coefficient),
                                         layers=round_repeats(1, depth_coefficient),
                                         stride=1,
                                         expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate)
        self.block2 = build_mbconv_block(in_channels=round_filters(16, width_coefficient),
                                         out_channels=round_filters(24, width_coefficient),
                                         layers=round_repeats(2, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
        self.block3 = build_mbconv_block(in_channels=round_filters(24, width_coefficient),
                                         out_channels=round_filters(40, width_coefficient),
                                         layers=round_repeats(2, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
        self.block4 = build_mbconv_block(in_channels=round_filters(40, width_coefficient),
                                         out_channels=round_filters(80, width_coefficient),
                                         layers=round_repeats(3, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
        self.block5 = build_mbconv_block(in_channels=round_filters(80, width_coefficient),
                                         out_channels=round_filters(112, width_coefficient),
                                         layers=round_repeats(3, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
        self.block6 = build_mbconv_block(in_channels=round_filters(112, width_coefficient),
                                         out_channels=round_filters(192, width_coefficient),
                                         layers=round_repeats(4, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
        self.block7 = build_mbconv_block(in_channels=round_filters(192, width_coefficient),
                                         out_channels=round_filters(320, width_coefficient),
                                         layers=round_repeats(1, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)

    def call(self, inputs, training=None, mask=None):
        features = []
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.swish(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        features.append(x)
        x = self.block4(x)
        features.append(x)
        x = self.block5(x)
        features.append(x)
        x = self.block6(x)
        features.append(x)
        x = self.block7(x)
        features.append(x)

        return features


def get_efficient_net(width_coefficient, depth_coefficient, dropout_rate):
    net = EfficientNet(width_coefficient=width_coefficient,
                       depth_coefficient=depth_coefficient,
                       dropout_rate=dropout_rate)

    return net


class BiFPN(tf.keras.layers.Layer):
    def __init__(self, output_channels, layers):
        super(BiFPN, self).__init__()
        self.levels = 5
        self.output_channels = output_channels
        self.layers = layers
        self.transform_convs = []
        self.bifpn_modules = []
        for _ in range(self.levels):
            self.transform_convs.append(ConvNormAct(filters=output_channels,
                                                    kernel_size=(1, 1),
                                                    strides=1,
                                                    padding="same"))
        for _ in range(self.layers):
            self.bifpn_modules.append(BiFPNModule(self.output_channels))

    def call(self, inputs, training=None, **kwargs):
        """
        :param inputs: list of features
        :param training:
        :param kwargs:
        :return: list of features
        """
        assert len(inputs) == self.levels
        x = []
        for i in range(len(inputs)):
            x.append(self.transform_convs[i](inputs[i], training=training))
        for j in range(self.layers):
            x = self.bifpn_modules[j](x, training=training)
        return x


class BiFPNModule(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(BiFPNModule, self).__init__()
        self.w_fusion_list = []
        self.conv_list = []
        for i in range(8):
            self.w_fusion_list.append(WeightedFeatureFusion(out_channels))
        self.upsampling_1 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.upsampling_2 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.upsampling_3 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.upsampling_4 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.maxpool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.maxpool_4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    def call(self, inputs, training=None, **kwargs):
        """
        :param inputs: list of features
        :param training:
        :param kwargs:
        :return:
        """
        assert len(inputs) == 5
        f3, f4, f5, f6, f7 = inputs
        f6_d = self.w_fusion_list[0]([f6, self.upsampling_1(f7)], training=training)
        f5_d = self.w_fusion_list[1]([f5, self.upsampling_2(f6_d)], training=training)
        f4_d = self.w_fusion_list[2]([f4, self.upsampling_3(f5_d)], training=training)

        f3_u = self.w_fusion_list[3]([f3, self.upsampling_4(f4_d)], training=training)
        f4_u = self.w_fusion_list[4]([f4, f4_d, self.maxpool_1(f3_u)], training=training)
        f5_u = self.w_fusion_list[5]([f5, f5_d, self.maxpool_2(f4_u)], training=training)
        f6_u = self.w_fusion_list[6]([f6, f6_d, self.maxpool_3(f5_u)], training=training)
        f7_u = self.w_fusion_list[7]([f7, self.maxpool_4(f6_u)], training=training)

        return [f3_u, f4_u, f5_u, f6_u, f7_u]


class SeparableConvNormAct(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 padding):
        super(SeparableConvNormAct, self).__init__()
        self.conv = tf.keras.layers.SeparableConv2D(filters=filters,
                                                    kernel_size=kernel_size,
                                                    strides=strides,
                                                    padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.swish(x)
        return x


class ConvNormAct(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 padding):
        super(ConvNormAct, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.swish(x)
        return x


class WeightedFeatureFusion(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(WeightedFeatureFusion, self).__init__()
        self.epsilon = 1e-4
        self.conv = SeparableConvNormAct(filters=out_channels, kernel_size=(3, 3), strides=1, padding="same")

    def build(self, input_shape):
        self.num_features = len(input_shape)
        assert self.num_features >= 2
        self.fusion_weights = self.add_weight(name="fusion_w",
                                              shape=(self.num_features, ),
                                              dtype=tf.dtypes.float32,
                                              initializer=tf.constant_initializer(value=1.0 / self.num_features),
                                              trainable=True)

    def call(self, inputs, training=None, **kwargs):
        """
        :param inputs: list of features
        :param kwargs:
        :return:
        """
        fusion_w = tf.nn.relu(self.fusion_weights)
        sum_features = []
        for i in range(self.num_features):
            sum_features.append(fusion_w[i] * inputs[i])
        output_feature = tf.reduce_sum(input_tensor=sum_features, axis=0) / (tf.reduce_sum(input_tensor=fusion_w) + self.epsilon)
        output_feature = self.conv(output_feature, training=training)
        return output_feature


class TransposeLayer(tf.keras.layers.Layer):
    def __init__(self, out_channels, num_layers=5):
        super(TransposeLayer, self).__init__()
        self.layers = num_layers
        self.transpose_layers = []
        for i in range(self.layers - 1):
            self.transpose_layers.append(tf.keras.Sequential([
                tf.keras.layers.Conv2DTranspose(filters=out_channels, kernel_size=(4, 4), strides=2, padding="same"),
                tf.keras.layers.BatchNormalization(),
            ]))

    def call(self, inputs, training=None, **kwargs):
        assert len(inputs) == self.layers
        f3, f4, f5, f6, f7 = inputs
        f6 += tf.nn.swish(self.transpose_layers[0](f7, training=training))
        f5 += tf.nn.swish(self.transpose_layers[1](f6, training=training))
        f4 += tf.nn.swish(self.transpose_layers[2](f5, training=training))
        f3 += tf.nn.swish(self.transpose_layers[3](f4, training=training))
        return f3


class EfficientDet(tf.keras.layers.Layer):
    def __init__(self, efficient_det):
        super(EfficientDet, self).__init__()
        self.heads = Config.heads
        self.head_conv = Config.head_conv[efficient_det]
        self.efficient_net = get_efficient_net(width_coefficient=Config.get_width_coefficient(efficient_det),
                                               depth_coefficient=Config.get_depth_coefficient(efficient_det),
                                               dropout_rate=Config.get_dropout_rate(efficient_det))
        self.bifpn = BiFPN(output_channels=Config.get_w_bifpn(efficient_det), layers=Config.get_d_bifpn(efficient_det))
        self.transpose = TransposeLayer(out_channels=Config.get_w_bifpn(efficient_det))
        for head in self.heads:
            classes = self.heads[head]
            if self.head_conv > 0:
                fc = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=self.head_conv, kernel_size=(3, 3), strides=1,
                                           padding="same", use_bias=True),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(filters=classes, kernel_size=(1, 1), strides=1,
                                           padding="same", use_bias=True)
                ])
            else:
                fc = tf.keras.layers.Conv2D(filters=classes, kernel_size=(1, 1), strides=1,
                                            padding="same", use_bias=True)
            self.__setattr__(head, fc)

    def call(self, inputs, training=None, **kwargs):
        x = self.efficient_net(inputs, training=training)
        x = self.bifpn(x, training=training)
        x = self.transpose(x, training=training)
        outputs = []
        for head in self.heads:
            outputs.append(self.__getattribute__(head)(x, training=training))
        return outputs


def d0():
    return EfficientDet("D0")


def d1():
    return EfficientDet("D1")


def d2():
    return EfficientDet("D2")


def d3():
    return EfficientDet("D3")


def d4():
    return EfficientDet("D4")


def d5():
    return EfficientDet("D5")


def d6():
    return EfficientDet("D6")


def d7():
    return EfficientDet("D7")
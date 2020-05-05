import tensorflow as tf

from configuration import Config


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=filter_num,
                                       kernel_size=(1, 1),
                                       strides=stride,
                                       use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])
        else:
            self.downsample = tf.keras.layers.Lambda(lambda x: x)

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs, training=training)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.downsample = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filter_num * 4,
                                   kernel_size=(1, 1),
                                   strides=stride,
                                   use_bias=False),
            tf.keras.layers.BatchNormalization()
        ])

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs, training=training)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output


class ResNetTypeI(tf.keras.layers.Layer):
    def __init__(self, layer_params, heads, head_conv):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = ResNetTypeI.__make_basic_block_layer(filter_num=64,
                                                           blocks=layer_params[0])
        self.layer2 = ResNetTypeI.__make_basic_block_layer(filter_num=128,
                                                           blocks=layer_params[1],
                                                           stride=2)
        self.layer3 = ResNetTypeI.__make_basic_block_layer(filter_num=256,
                                                           blocks=layer_params[2],
                                                           stride=2)
        self.layer4 = ResNetTypeI.__make_basic_block_layer(filter_num=512,
                                                           blocks=layer_params[3],
                                                           stride=2)
        self.transposed_conv_layers = ResNetTypeI.__make_transposed_conv_layer(num_layers=3,
                                                                               num_filters=[256, 256, 256],
                                                                               num_kernels=[4, 4, 4])

        if head_conv > 0:
            self.heatmap_layer = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=head_conv, kernel_size=(3, 3), strides=1, padding="same"),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=heads["heatmap"], kernel_size=(1, 1), strides=1, padding="same")
            ])
            self.reg_layer = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=head_conv, kernel_size=(3, 3), strides=1, padding="same"),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=heads["reg"], kernel_size=(1, 1), strides=1, padding="same")
            ])
            self.wh_layer = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=head_conv, kernel_size=(3, 3), strides=1, padding="same"),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=heads["wh"], kernel_size=(1, 1), strides=1, padding="same")
            ])
        else:
            self.heatmap_layer = tf.keras.layers.Conv2D(filters=heads["heatmap"], kernel_size=(1, 1), strides=1, padding="same")
            self.reg_layer = tf.keras.layers.Conv2D(filters=heads["reg"], kernel_size=(1, 1), strides=1, padding="same")
            self.wh_layer = tf.keras.layers.Conv2D(filters=heads["wh"], kernel_size=(1, 1), strides=1, padding="same")

    @staticmethod
    def __make_basic_block_layer(filter_num, blocks, stride=1):
        res_block = tf.keras.Sequential()
        res_block.add(BasicBlock(filter_num, stride=stride))

        for _ in range(1, blocks):
            res_block.add(BasicBlock(filter_num, stride=1))

        return res_block

    @staticmethod
    def __make_transposed_conv_layer(num_layers, num_filters, num_kernels):
        layers = tf.keras.Sequential()
        for i in range(num_layers):
            layers.add(tf.keras.layers.Conv2DTranspose(filters=num_filters[i],
                                                       kernel_size=num_kernels[i],
                                                       strides=2,
                                                       padding="same",
                                                       use_bias=False))
            layers.add(tf.keras.layers.BatchNormalization())
            layers.add(tf.keras.layers.ReLU())
        return layers

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.transposed_conv_layers(x, training=training)
        heatmap = self.heatmap_layer(x, training=training)
        reg = self.reg_layer(x, training=training)
        wh = self.wh_layer(x, training=training)

        return [heatmap, reg, wh]


class ResNetTypeII(tf.keras.layers.Layer):
    def __init__(self, layer_params, heads, head_conv):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = ResNetTypeII.__make_bottleneck_layer(filter_num=64,
                                                           blocks=layer_params[0])
        self.layer2 = ResNetTypeII.__make_bottleneck_layer(filter_num=128,
                                                           blocks=layer_params[1],
                                                           stride=2)
        self.layer3 = ResNetTypeII.__make_bottleneck_layer(filter_num=256,
                                                           blocks=layer_params[2],
                                                           stride=2)
        self.layer4 = ResNetTypeII.__make_bottleneck_layer(filter_num=512,
                                                           blocks=layer_params[3],
                                                           stride=2)

        self.transposed_conv_layers = ResNetTypeII.__make_transposed_conv_layer(num_layers=3,
                                                                                num_filters=[256, 256, 256],
                                                                                num_kernels=[4, 4, 4])

        if head_conv > 0:
            self.heatmap_layer = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=head_conv, kernel_size=(3, 3), strides=1, padding="same"),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=heads["heatmap"], kernel_size=(1, 1), strides=1, padding="same")
            ])
            self.reg_layer = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=head_conv, kernel_size=(3, 3), strides=1, padding="same"),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=heads["reg"], kernel_size=(1, 1), strides=1, padding="same")
            ])
            self.wh_layer = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=head_conv, kernel_size=(3, 3), strides=1, padding="same"),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=heads["wh"], kernel_size=(1, 1), strides=1, padding="same")
            ])
        else:
            self.heatmap_layer = tf.keras.layers.Conv2D(filters=heads["heatmap"], kernel_size=(1, 1), strides=1, padding="same")
            self.reg_layer = tf.keras.layers.Conv2D(filters=heads["reg"], kernel_size=(1, 1), strides=1, padding="same")
            self.wh_layer = tf.keras.layers.Conv2D(filters=heads["wh"], kernel_size=(1, 1), strides=1, padding="same")

    @staticmethod
    def __make_bottleneck_layer(filter_num, blocks, stride=1):
        res_block = tf.keras.Sequential()
        res_block.add(BottleNeck(filter_num, stride=stride))

        for _ in range(1, blocks):
            res_block.add(BottleNeck(filter_num, stride=1))

        return res_block

    @staticmethod
    def __make_transposed_conv_layer(num_layers, num_filters, num_kernels):
        layers = tf.keras.Sequential()
        for i in range(num_layers):
            layers.add(tf.keras.layers.Conv2DTranspose(filters=num_filters[i],
                                                       kernel_size=num_kernels[i],
                                                       strides=2,
                                                       padding="same",
                                                       use_bias=False))
            layers.add(tf.keras.layers.BatchNormalization())
            layers.add(tf.keras.layers.ReLU())
        return layers

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.transposed_conv_layers(x, training=training)
        heatmap = self.heatmap_layer(x, training=training)
        reg = self.reg_layer(x, training=training)
        wh = self.wh_layer(x, training=training)

        return [heatmap, reg, wh]


def resnet_18():
    return ResNetTypeI(layer_params=[2, 2, 2, 2], heads=Config.heads, head_conv=Config.head_conv["resnets"])


def resnet_34():
    return ResNetTypeI(layer_params=[3, 4, 6, 3], heads=Config.heads, head_conv=Config.head_conv["resnets"])


def resnet_50():
    return ResNetTypeII(layer_params=[3, 4, 6, 3], heads=Config.heads, head_conv=Config.head_conv["resnets"])


def resnet_101():
    return ResNetTypeII(layer_params=[3, 4, 23, 3], heads=Config.heads, head_conv=Config.head_conv["resnets"])


def resnet_152():
    return ResNetTypeII(layer_params=[3, 8, 36, 3], heads=Config.heads, head_conv=Config.head_conv["resnets"])

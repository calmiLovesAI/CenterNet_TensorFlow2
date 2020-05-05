import tensorflow as tf
import numpy as np
from core.models.group_convolution import GroupConv2D, GroupConv2DTranspose
from configuration import Config


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(3, 3), strides=stride,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(3, 3), strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, residual=None, **kwargs):
        if residual is None:
            residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        outputs = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return outputs


class BottleNeck(tf.keras.layers.Layer):
    expansion = 2
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        temp_channels = out_channels // BottleNeck.expansion
        self.conv1 = tf.keras.layers.Conv2D(filters=temp_channels, kernel_size=(1, 1), strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=temp_channels, kernel_size=(3, 3), strides=stride,
                                            padding="same",
                                            use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(1, 1), strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, residual=None, **kwargs):
        if residual is None:
            residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        outputs = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return outputs


class BottleNeckX(tf.keras.layers.Layer):
    cardinality = 32
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeckX, self).__init__()
        temp_channels = out_channels * BottleNeckX.cardinality // 32
        self.conv1 = GroupConv2D(input_channels=in_channels, output_channels=temp_channels,
                                 kernel_size=(1, 1), strides=1, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = GroupConv2D(input_channels=temp_channels, output_channels=temp_channels,
                                 kernel_size=(3, 3), strides=stride, padding="same", use_bias=False,
                                 groups=BottleNeckX.cardinality)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = GroupConv2D(input_channels=temp_channels, output_channels=out_channels,
                                 kernel_size=(1, 1), strides=1, padding="same", use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, residual=None, **kwargs):
        if residual is None:
            residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        outputs = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return outputs


class Root(tf.keras.layers.Layer):
    def __init__(self, out_channels, residual):
        super(Root, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(1, 1), padding="same",
                                           strides=1, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.residual = residual

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(tf.concat(values=inputs, axis=-1))
        x = self.bn(x, training=training)
        if self.residual:
            x = tf.keras.layers.add([x, inputs[0]])
        x = tf.nn.relu(x)
        return x


class Tree(tf.keras.layers.Layer):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride)
            self.tree2 = block(out_channels, out_channels, 1)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              root_residual=root_residual)
        if levels == 1:
            self.root = Root(out_channels, root_residual)

        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels

        if stride > 1:
            self.downsample = tf.keras.layers.MaxPool2D(pool_size=stride, strides=stride, padding="same")
        if in_channels != out_channels:
            self.project = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(1, 1), strides=1, padding="same",
                                       use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])

    def call(self, inputs, training=None, residual=None, children=None, **kwargs):
        children = [] if children is None else children
        bottom = self.downsample(inputs) if self.downsample else inputs
        residual = self.project(bottom, training=training) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(inputs, training=training, residual=residual)
        if self.levels == 1:
            x2 = self.tree2(x1, training=training)
            outputs = self.root([x2, x1, *children], training=training)
        else:
            children.append(x1)
            outputs = self.tree2(x1, training=training, children=children)
        return outputs


class DLA(tf.keras.layers.Layer):
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock,
                 residual_root=False, return_levels=False, pool_size=7):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes

        self.base_layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=channels[0], kernel_size=7, strides=1, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.level_0 = DLA.__make_conv_level(out_channels=channels[0], convs=levels[0])
        self.level_1 = DLA.__make_conv_level(out_channels=channels[1], convs=levels[1], stride=2)
        self.level_2 = Tree(levels=levels[2], block=block, in_channels=channels[1],
                            out_channels=channels[2], stride=2,
                            level_root=False, root_residual=residual_root)
        self.level_3 = Tree(levels=levels[3], block=block, in_channels=channels[2],
                            out_channels=channels[3], stride=2,
                            level_root=True, root_residual=residual_root)
        self.level_4 = Tree(levels=levels[4], block=block, in_channels=channels[3],
                            out_channels=channels[4], stride=2,
                            level_root=True, root_residual=residual_root)
        self.level_5 = Tree(levels=levels[5], block=block, in_channels=channels[4],
                            out_channels=channels[5], stride=2,
                            level_root=True, root_residual=residual_root)

        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=pool_size)
        self.final = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=1,
                                            padding="same", use_bias=True)

    @staticmethod
    def __make_conv_level(out_channels, convs, stride=1):
        layers = []
        for i in range(convs):
            if i == 0:
                layers.extend([tf.keras.layers.Conv2D(filters=out_channels,
                                                      kernel_size=(3, 3),
                                                      strides=stride,
                                                      padding="same",
                                                      use_bias=False),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.ReLU()])
            else:
                layers.extend([tf.keras.layers.Conv2D(filters=out_channels,
                                                      kernel_size=(3, 3),
                                                      strides=1,
                                                      padding="same",
                                                      use_bias=False),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.ReLU()])
        return tf.keras.Sequential(layers)

    def call(self, inputs, training=None, **kwargs):
        y = []
        x = self.base_layer(inputs, training=training)

        x = self.level_0(x, training=training)
        y.append(x)
        x = self.level_1(x, training=training)
        y.append(x)
        x = self.level_2(x, training=training)
        y.append(x)
        x = self.level_3(x, training=training)
        y.append(x)
        x = self.level_4(x, training=training)
        y.append(x)
        x = self.level_5(x, training=training)
        y.append(x)

        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.final(x)
            x = tf.reshape(x, (x.shape[0], -1))
            return x


class Identity(tf.keras.layers.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, inputs, **kwargs):
        return inputs


class IDAUp(tf.keras.layers.Layer):
    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=out_dim, kernel_size=(1, 1), strides=1, padding="same",
                                           use_bias=False),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU()
                ])
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = GroupConv2DTranspose(input_channels=out_dim, output_channels=out_dim, kernel_size=f * 2,
                                          strides=f, padding="same", groups=out_dim, use_bias=False)
            setattr(self, "proj_" + str(i), proj)
            setattr(self, "up_" + str(i), up)

        for i in range(1, len(channels)):
            node = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=out_dim, kernel_size=node_kernel, strides=1,
                                       padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ])
            setattr(self, "node_" + str(i), node)

    def call(self, inputs, training=None, **kwargs):
        layers = list(inputs)
        for i, l in enumerate(layers):
            upsample = getattr(self, "up_" + str(i))
            project = getattr(self, "proj_" + str(i))
            layers[i] = upsample(project(l, training=training))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, "node_" + str(i))
            x = node(tf.concat([x, layers[i]], -1))
            y.append(x)
        return x, y


class DLAUp(tf.keras.layers.Layer):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=np.int32)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(3, channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def call(self, inputs, training=None, **kwargs):
        layers = list(inputs)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:], training=training)
            layers[-i - 1:] = y
        return x


class DLASeg(tf.keras.layers.Layer):
    def __init__(self, base_name, heads, down_ratio=4, head_conv=256):
        super(DLASeg, self).__init__()
        self.heads = heads
        self.first_level = int(np.log2(down_ratio))
        self.base = DLASeg.__get_base_block(base_name)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=head_conv, kernel_size=(3, 3), strides=1,
                                           padding="same", use_bias=True),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(filters=classes, kernel_size=(1, 1), strides=1,
                                           padding="same", use_bias=True)
                ])
            else:
                fc = tf.keras.layers.Conv2D(filters=classes, kernel_size=(1, 1), strides=1,
                                            padding="same", use_bias=True)
            self.__setattr__(head, fc)


    @staticmethod
    def __get_base_block(base_name):
        if base_name == "dla34":
            return DLA(levels=[1, 1, 1, 2, 2, 1], channels=[16, 32, 64, 128, 256, 512], block=BasicBlock,
                       return_levels=True)
        elif base_name == "dla60":
            return DLA(levels=[1, 1, 1, 2, 3, 1], channels=[16, 32, 128, 256, 512, 1024], block=BottleNeck,
                       return_levels=True)
        elif base_name == "dla102":
            return DLA(levels=[1, 1, 1, 3, 4, 1], channels=[16, 32, 128, 256, 512, 1024], block=BottleNeck,
                       residual_root=True, return_levels=True)
        elif base_name == "dla169":
            return DLA(levels=[1, 1, 2, 3, 5, 1], channels=[16, 32, 128, 256, 512, 1024], block=BottleNeck,
                       residual_root=True, return_levels=True)
        else:
            raise ValueError("The 'base_name' is invalid.")

    def call(self, inputs, training=None, **kwargs):
        x = self.base(inputs, training=training)
        x = self.dla_up(x[self.first_level:], training=training)
        outputs = []
        for head in self.heads:
            outputs.append(self.__getattribute__(head)(x, training=training))
        return outputs


def dla_34():
    return DLASeg(base_name="dla34", heads=Config.heads, down_ratio=Config.downsampling_ratio,
                  head_conv=Config.head_conv["dla"])


def dla_60():
    return DLASeg(base_name="dla60", heads=Config.heads, down_ratio=Config.downsampling_ratio,
                  head_conv=Config.head_conv["dla"])


def dla_102():
    return DLASeg(base_name="dla102", heads=Config.heads, down_ratio=Config.downsampling_ratio,
                  head_conv=Config.head_conv["dla"])


def dla_169():
    return DLASeg(base_name="dla169", heads=Config.heads, down_ratio=Config.downsampling_ratio,
                  head_conv=Config.head_conv["dla"])

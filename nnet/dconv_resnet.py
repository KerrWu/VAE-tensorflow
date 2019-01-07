# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet model.
Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import six

from tensorflow.python.training import moving_averages

HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')

tf.logging.warning("models/research/resnet is deprecated. "
                   "Please use models/official/resnet instead.")


def weight_init(shape, name=None, initializer=tf.contrib.layers.xavier_initializer()):
    """
    Weights Initialization
    """

    if name is None:
        name = 'W'

    W = tf.get_variable(name=name, shape=shape,
                        initializer=initializer)

    tf.summary.histogram(name, W)
    return W


def bias_init(shape, name=None, constant=0.0):
    """
    Bias Initialization
    """

    if name is None:
        name = 'b'

    b = tf.get_variable(name=name, shape=shape,
                        initializer=tf.constant_initializer(constant))

    tf.summary.histogram(name, b)
    return b


def leaky_relu(input, alpha=0.2, name="lrelu"):
    """
    Leaky ReLU
    """
    with tf.variable_scope(name):
        o1 = 0.5 * (1 + alpha)
        o2 = 0.5 * (1 - alpha)
        return o1 * input + o2 * abs(input)


def fully_connected_linear(input, output, name=None, reuse=False,
                           bias_constant=0.0, initializer=tf.contrib.layers.xavier_initializer()):
    """
    Fully-connected linear activations
    """

    if name is None:
        name = 'fully_connected_linear'

    shape = input.get_shape()
    input_units = int(shape[1])

    W = weight_init([input_units, output], 'W', initializer)
    b = bias_init([output], 'b', bias_constant)

    output = tf.add(tf.matmul(input, W), b)
    return output


def fully_connected(input, output, is_training, activation=tf.nn.relu,
                    name=None, use_batch_norm=False, reuse=False, use_leak=False, alpha=0.2,
                    initializer=tf.contrib.layers.xavier_initializer(), bias_constant=0.0):
    """
    Fully-connected layer with induced non-linearity of 'relu'
    """

    if name is None:
        name = 'fully_connected'

    with tf.variable_scope(name, reuse=reuse):
        output = fully_connected_linear(input=input, output=output, name=name, reuse=reuse,
                                        initializer=initializer, bias_constant=bias_constant)

        if activation is None:
            return output
        else:
            if use_batch_norm:
                if use_leak:
                    return leaky_relu(batch_norm(output, is_training=is_training), alpha)
                else:
                    return activation(batch_norm(output, is_training=is_training))
            else:
                if use_leak:
                    return leaky_relu(output, alpha)
                else:
                    return activation(output)


class ResNetDeconv(object):
    """ResNet model.
    conv2d_transpose(value,
    filter,
    output_shape,
    strides,
    padding="SAME",
    data_format="NHWC",
    name=None)
    第一个参数value：指需要做反卷积的输入图像，它要求是一个Tensor
    第二个参数filter：卷积核，它要求是一个Tensor，具有[filter_height, filter_width, out_channels, in_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，卷积核个数，图像通道数]
    第三个参数output_shape：反卷积操作输出的shape，细心的同学会发现卷积操作是没有这个参数的，那这个参数在这里有什么用呢？下面会解释这个问题
    第四个参数strides：反卷积时在图像每一维的步长，这是一个一维的向量，长度4
    第五个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
    第六个参数data_format：string类型的量，'NHWC'和'NCHW'其中之一，这是tensorflow新版本中新加的参数，它说明了value参数的数据格式。'NHWC'指tensorflow标准的数据格式[batch, height, width, in_channels]，'NCHW'指Theano的数据格式,[batch, in_channels，height, width]，当然默认值是'NHWC'
    """

    def __init__(self, hps, images, mode, opts, dims):
        """ResNet constructor.
        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          mode: One of 'train' and 'eval'.
        """
        self.hps = hps
        self._images = images
        self.mode = mode
        if mode == 'train':
            self.is_training = True
        else:
            self.is_training = False

        self._extra_train_ops = []
        self.opts = opts
        self.dims = dims

    def build_graph(self):
        """Build a whole graph for the model."""
        # self.global_step = tf.train.get_or_create_global_step()
        self._build_model()
        # self.summaries = tf.summary.merge_all()

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d_transpose."""
        return [1, stride, stride, 1]

    def _build_model(self):
        """Build the core model within the graph."""
        with tf.variable_scope('init'):
            x = self._images
            x = self._dconv(x, [1, 1, 1024, 1024],
                            [self.hps.batch_size, 5, 5, 1024],
                            stride=1, name="init",
                            activation=None,
                            initializer=tf.truncated_normal_initializer(stddev=0.02),
                            use_leak=True, alpha=self.hps.relu_leakiness, use_batch_norm=False)
            print("x init shape", x.get_shape())

        #strides = [2, 2, 2, 2, 2, 2, 1]
        strides = [1, 2, 2, 2, 2, 2, 2, 1]
        activate_before_residual = [False, False, False, False, False, False, True]
        dims_list = [32, 16, 8, 4, 2, 1, 1]
        map_size = [5, 10, 19, 38, 75, 150, 300]
        if self.hps.use_bottleneck:
            res_func = self._bottleneck_residual
            filters = [16, 64, 128, 256]
        else:
            res_func = self._residual
            filters = [1024, 1024, 512, 128, 64, 32, 16, 8]
            # Uncomment the following codes to use w28-10 wide residual network.
            # It is more memory efficient than very deep residual network and has
            # comparably good performance.
            # https://arxiv.org/pdf/1605.07146v1.pdf
            # filters = [16, 160, 320, 640]
            # Update hps.num_residual_units to 4

            # def _residual(self, x, in_filter, out_filter, stride, dims, map_size,
            #                   activate_before_residual=False, is_training=True):

        with tf.variable_scope('dconv_unit_1_0'):
            print("building dconv unit 1")
            x = res_func(x, filters[0], filters[1],
                         strides[0],
                         (self.dims * dims_list[0]),
                         map_size[0],
                         activate_before_residual[0], is_training=self.is_training)

            print('dconv_unit_1_0', x.get_shape())

        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('dconv_unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1],
                             1,
                             (self.dims * dims_list[0]),
                             map_size[0],
                             False, is_training=self.is_training)
                print('dconv_unit_1_%d' % i, x.get_shape())

        with tf.variable_scope('dconv_unit_2_0'):
            x = res_func(x, filters[1], filters[2], strides[1],
                         (self.dims * dims_list[1]),
                         map_size[1],
                         activate_before_residual[1], is_training=self.is_training)
            print('dconv_unit_2_0', x.get_shape())

        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('dconv_unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2],
                             1,
                             (self.dims * dims_list[1]),
                             map_size[1],
                             False, is_training=self.is_training)
                print('dconv_unit_2_%d' % i, x.get_shape())

        with tf.variable_scope('dconv_unit_3_0'):
            x = res_func(x, filters[2], filters[3], strides[2],
                         (self.dims * dims_list[2]),
                         map_size[2],
                         activate_before_residual[2], is_training=self.is_training)
            print('dconv_unit_2_0', x.get_shape())

        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('dconv_unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], 1,
                             (self.dims * dims_list[2]),
                             map_size[2],
                             False, is_training=self.is_training)
                print('dconv_unit_3_%d' % i, x.get_shape())

        with tf.variable_scope('dconv_unit_4_0'):
            x = res_func(x, filters[3], filters[4], strides[3],
                         (self.dims * dims_list[3]),
                         map_size[3],
                         activate_before_residual[3], is_training=self.is_training)
            print('dconv_unit_2_0', x.get_shape())

        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('dconv_unit_4_%d' % i):
                x = res_func(x, filters[4], filters[4], 1,
                             (self.dims * dims_list[3]),
                             map_size[3],
                             False, is_training=self.is_training)
                print('dconv_unit_4_%d' % i, x.get_shape())

        with tf.variable_scope('dconv_unit_5_0'):
            x = res_func(x, filters[4], filters[5], strides[4],
                         (self.dims * dims_list[4]),
                         map_size[4],
                         activate_before_residual[4], is_training=self.is_training)
            print('dconv_unit_5_0', x.get_shape())

        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('dconv_unit_5_%d' % i):
                x = res_func(x, filters[5], filters[5], 1,
                             (self.dims * dims_list[4]),
                             map_size[4],
                             False, is_training=self.is_training)
                print('dconv_unit_5_%d' % i, x.get_shape())

        with tf.variable_scope('dconv_unit_6_0'):
            x = res_func(x, filters[5], filters[6], strides[5],
                         (self.dims * dims_list[5]),
                         map_size[5],
                         activate_before_residual[5], is_training=self.is_training)
            print('dconv_unit_6_0', x.get_shape())

        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('dconv_unit_6_%d' % i):
                x = res_func(x, filters[6], filters[6], 1,
                             (self.dims * dims_list[5]),
                             map_size[5],
                             False, is_training=self.is_training)
                print('dconv_unit_6_%d' % i, x.get_shape())
                
        with tf.variable_scope('dconv_unit_7_0'):
            x = res_func(x, filters[6], filters[7], strides[6],
                         (self.dims * dims_list[6]),
                         map_size[6],
                         activate_before_residual[6], is_training=self.is_training)
            print('dconv_unit_7_0', x.get_shape())

        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('dconv_unit_7_%d' % i):
                x = res_func(x, filters[7], filters[7], 1,
                             (self.dims * dims_list[6]),
                             map_size[6],
                             False, is_training=self.is_training)
                print('dconv_unit_7_%d' % i, x.get_shape())

        with tf.variable_scope('dconv_unit_last'):
            x = leaky_relu(batch_norm(x, is_training=self.is_training), self.hps.relu_leakiness)
            print('dconv_unit_last', x.get_shape())

        with tf.variable_scope('dconv_logit'):
            self.dconv_out = x

    # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py

    def _residual(self, x, in_filter, out_filter, stride, dims, map_size,
                  activate_before_residual=False, is_training=True):
        """Residual unit with 2 sub layers."""
        # def deconv(input, kernel, output_shape, stride=1, name=None,
        #    activation=None, use_batch_norm=False, is_training=False,
        #    reuse=False, initializer=tf.contrib.layers.xavier_initializer(),
        #    bias_constant=0.0, use_leak=False, alpha=0.0):

        # dconv2 = model.deconv(x, [3, 3, dims * (16), dims * (32)],
        #                       [self.opts.batch_size, 4, 4, dims * (16)], 2, "dconv2", tf.nn.relu,
        #                       initializer=tf.truncated_normal_initializer(stddev=0.02), use_leak=True,
        #                       alpha=0.2)  # 8x8x128/ 8*8*256
        out_filter = int(out_filter // 2)
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = batch_norm(x, is_training=is_training)
                x = leaky_relu(x, self.hps.relu_leakiness)
                orig_x = self._dconv(x, [1, 1, out_filter, in_filter],
                                     [self.hps.batch_size, map_size, map_size, out_filter],
                                     stride, "dconv3",
                                     None,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     use_leak=True, alpha=self.hps.relu_leakiness, use_batch_norm=True)
        else:

            with tf.variable_scope('residual_only_activation'):
                orig_x = self._dconv(x, [1, 1, out_filter, in_filter],
                                     [self.hps.batch_size, map_size, map_size, out_filter],
                                     stride, "dconv3",
                                     None,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     use_leak=True, alpha=self.hps.relu_leakiness, use_batch_norm=True)
                x = batch_norm(x, is_training=is_training)
                x = leaky_relu(x, self.hps.relu_leakiness)

        # out_filter = int(out_filter // 2)
        with tf.variable_scope('sub1'):
            x = self._dconv(x, [3, 3, out_filter, in_filter],
                            [self.hps.batch_size, map_size, map_size, out_filter],
                            stride, "dconv3",
                            tf.nn.relu,
                            initializer=tf.truncated_normal_initializer(stddev=0.02),
                            use_leak=True, alpha=self.hps.relu_leakiness, use_batch_norm=True)

        with tf.variable_scope('sub2'):
            x = self._dconv(x, [3, 3, out_filter, out_filter],
                            [self.hps.batch_size, map_size, map_size, out_filter],
                            1, "dconv3",
                            None,
                            initializer=tf.truncated_normal_initializer(stddev=0.02),
                            use_leak=False, alpha=self.hps.relu_leakiness, use_batch_norm=False)

            #        with tf.variable_scope('sub_add'):
            #            if in_filter != out_filter:
            #                print("orig shape = ",orig_x.get_shape(),"outshape = ",x.get_shape())
            #                orig_x = self._dconv(orig_x, [3,3,out_filter,in_filter],
            #                            [self.opts.batch_size, map_size, map_size, out_filter],
            #                            stride,"dconv3",
            #                            None,
            #                            initializer=tf.truncated_normal_initializer(stddev=0.02),
            #                            use_leak=False, alpha=self.hps.relu_leakiness,use_batch_norm=False)
            #                print("processed orig shape = ",orig_x.get_shape(),"outshape = ",x.get_shape())

            # x += orig_x
            # print("orig shape = ",orig_x.get_shape(),"outshape = ",x.get_shape())
            x = tf.concat([x, orig_x], 3)
            # print("concated shape", x.get_shape())

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                             activate_before_residual=False):
        """Bottleneck residual unit with 3 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('common_bn_relu'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_bn_relu'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 1, in_filter, out_filter / 4, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

        with tf.variable_scope('sub3'):
            x = self._batch_norm('bn3', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv3', x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
            x += orig_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _dconv(self, input, kernel, output_shape, stride=1, name=None,
               activation=None, use_batch_norm=False, is_training=False,
               reuse=False, initializer=tf.contrib.layers.xavier_initializer(),
               bias_constant=0.0, use_leak=False, alpha=0.0):
        """
        2D convolution layer with relu activation
        """

        if name is None:
            name = 'de_convolution'

        with tf.variable_scope(name, reuse):
            W = weight_init(kernel, 'W', initializer)
            b = bias_init(kernel[2], 'b', bias_constant)

            strides = [1, stride, stride, 1]
            output = tf.nn.conv2d_transpose(value=input, filter=W, output_shape=output_shape, strides=strides)
            output = output + b

            if activation is None:
                return output

            if use_batch_norm:
                if use_leak:
                    return leaky_relu(batch_norm(output, is_training=is_training), alpha)
                else:
                    return activation(batch_norm(output, is_training=is_training))
            else:
                if use_leak:
                    return leaky_relu(output, alpha)
                else:
                    return activation(output)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

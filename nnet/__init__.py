import tensorflow as tf
import numpy as np
import time
import datetime
import os
import re
import math

from nnet import modules as model
from nnet import resnet
from nnet import dconv_resnet
import utils


class GAN(object):
    def __init__(self, opts, is_training=False):
        self.h = opts.image_size_h
        self.w = opts.image_size_w
        self.c = opts.channels
        self.opts = opts
        self.images = tf.placeholder(tf.float32, [None, self.h, self.w, self.c], "images")
        if self.opts.use_labels:
            self.labels = tf.placeholder(tf.float32, [None, self.opts.label_len], "labels")
        self.code = tf.placeholder(tf.float32, [None, self.opts.code_len], "code")
        self.D_lr = tf.placeholder(tf.float32, [], "D_learning_rate")
        self.G_lr = tf.placeholder(tf.float32, [], "G_learning_rate")
        self.is_training = self.opts.train
        self.generated_imgs = self.Generator(self.code)
        self.true_logit = self.Discriminator(self.images, reuse=False)
        self.fake_logit = self.Discriminator(self.generated_imgs, reuse=True)
        self.d_loss, self.g_loss = self.loss()

        # self.plot_gradients_summary("Discriminator")
        # self.plot_gradients_summary("Generator")

        d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]

        self.sess = tf.Session()
        with tf.variable_scope("Optimizers"):
            self.D_optimizer = tf.train.AdamOptimizer(self.D_lr).minimize(self.d_loss, var_list=d_vars)
            self.G_optimizer = tf.train.AdamOptimizer(self.G_lr).minimize(self.g_loss, var_list=g_vars)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        tf.summary.scalar('Discriminator loss: ', self.d_loss)
        tf.summary.scalar('Generator loss', self.g_loss)
        tf.summary.scalar('Discriminator Learning Rate', self.D_lr)
        tf.summary.scalar('Generator Learning Rate', self.G_lr)
        tf.summary.image('Generated image', self.generated_imgs, max_outputs=4)
        self.summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.opts.root_dir + self.opts.summary_dir, self.sess.graph)

    def plot_gradients_summary(self, name):
        if name == "Discriminator":
            d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
            [tf.summary.histogram(name + '_grad' + '/{}'.format(var.name), tf.gradients(self.d_loss, var)) for var in
             d_vars]
        else:
            g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
            [tf.summary.histogram(name + '_grad' + '/{}'.format(var.name), tf.gradients(self.g_loss, var)) for var in
             g_vars]

    def Discriminator(self, data, reuse=False):
        """
        Discriminator part of GAN
        """

        dims = self.opts.dims
        if self.opts.dataset == "CIFAR":
            with tf.variable_scope("discriminator"):
                conv1 = model.conv2d(data, [5, 5, 3, self.dims], 2, "conv1", is_training, False, reuse=reuse)
                conv2 = model.conv2d(conv1, [3, 3, self.dims, self.dims * 2], 2, "conv2", is_training, True,
                                     reuse=reuse, use_batch_norm=True)
                conv3 = model.conv2d(conv2, [3, 3, self.dims * 2, self.dims * 4], 2, "conv3", is_training, True,
                                     reuse=reuse, use_batch_norm=True)
                full4 = model.fully_connected(tf.reshape(conv3, [self.opts.batch_size, -1]), self.opts.label_len,
                                              is_training, None, "full4", False, reuse=reuse)
                return full4
        else:
            with tf.variable_scope("discriminator"):
                conv1 = model.conv2d(data, [5, 5, self.c, dims], 2, "conv1", alpha=0.2, use_leak=True,
                                     bias_constant=0.01, reuse=reuse, use_batch_norm=False,
                                     is_training=self.is_training)  # 14x14x64
                conv2 = model.conv2d(conv1, [5, 5, dims, dims * 2], 2, "conv2", alpha=0.2, use_leak=True,
                                     bias_constant=0.01, reuse=reuse, use_batch_norm=False,
                                     is_training=self.is_training)  # 7x7x128
                # conv2_flat = tf.reshape(conv2, [-1, int(np.prod(conv2.get_shape()[1:]))])
                conv3 = model.conv2d(conv2, [3, 3, dims * 2, dims * 4], 2, "conv3", alpha=0.2, use_leak=True,
                                     bias_constant=0.01, reuse=reuse, use_batch_norm=True,
                                     is_training=self.is_training)  # 4x4x256
                full1 = model.fully_connected(tf.reshape(conv3, [-1, 4 * 4 * dims * 4]), dims * 4 * 4 * 2,
                                              activation=tf.nn.relu, use_leak=True, name="full1", bias_constant=0.01,
                                              reuse=reuse, use_batch_norm=True, is_training=self.is_training)  # 1
                full2 = model.fully_connected(full1, dims * 4 * 4, activation=tf.nn.relu, name="full2",
                                              bias_constant=0.01, reuse=reuse, use_leak=True, use_batch_norm=True,
                                              is_training=self.is_training)  # 1
                output = model.fully_connected(full2, self.opts.label_len, activation=None, name="output",
                                               bias_constant=0.01, reuse=reuse, use_leak=True, use_batch_norm=True,
                                               is_training=self.is_training)  # 1

                # output = model.fully_connected(conv2_flat, 1, activation=None, use_leak=False, name="full1", bias_constant=0.01, reuse=reuse, use_batch_norm=False, is_training=self.is_training) # 1
                return output

    def Generator(self, code, reuse=False):
        """
        Generator part of GAN
        """

        dims = self.opts.dims
        if self.opts.dataset == "CIFAR":
            with tf.variable_scope("generator"):
                full1 = model.fully_connected(code, dims * 4 * 4 * 4, is_training, tf.nn.relu, "full1", False,
                                              reuse=reuse, use_batch_norm=True)
                dconv2 = model.deconv(tf.reshape(full1, [-1, 4, 4, dims * 4]), [8, 8, dims * 2, dims * 4], 2, "dconv2",
                                      is_training, False, reuse=reuse, use_batch_norm=True)
                dconv3 = model.deconv(dconv2, [16, 16, dims, dims * 2], 2, "dconv3", is_training, False, reuse=reuse,
                                      use_batch_norm=True)
                dconv4 = model.deconv(dconv3, [32, 32, dims, 3], 2, "dconv4", is_training, False, reuse=reuse)
                return tf.nn.tanh(dconv4)
        else:
            with tf.variable_scope("generator"):
                full1 = model.fully_connected(code, 7 * 7 * dims * 2, is_training=self.is_training,
                                              activation=tf.nn.relu, name="full1", reuse=reuse, bias_constant=0.01,
                                              use_leak=True, use_batch_norm=True,
                                              initializer=tf.truncated_normal_initializer(stddev=0.2))
                full2 = model.fully_connected(full1, 4 * 4 * dims * 2, is_training=self.is_training,
                                              activation=tf.nn.relu, name="full2", reuse=reuse, bias_constant=0.01,
                                              use_leak=True, use_batch_norm=True,
                                              initializer=tf.truncated_normal_initializer(stddev=0.2))
                full3 = model.fully_connected(full2, 4 * 4 * dims * 4, is_training=self.is_training,
                                              activation=tf.nn.relu, name="full3", reuse=reuse, bias_constant=0.01,
                                              use_leak=True, use_batch_norm=True,
                                              initializer=tf.truncated_normal_initializer(stddev=0.2))
                dconv2 = model.deconv(tf.reshape(full3, [-1, 4, 4, dims * 4]), [3, 3, dims * 2, dims * 4],
                                      [self.opts.batch_size, 7, 7, dims * 2], 2, "dconv2", tf.nn.relu,
                                      initializer=tf.truncated_normal_initializer(stddev=0.2),
                                      bias_constant=0.0, reuse=reuse, use_batch_norm=True, is_training=self.is_training,
                                      use_leak=True)
                dconv3 = model.deconv(dconv2, [3, 3, dims, dims * 2], [self.opts.batch_size, 14, 14, dims], 2, "dconv3",
                                      tf.nn.relu, initializer=tf.truncated_normal_initializer(stddev=0.2), \
                                      bias_constant=0.0, reuse=reuse, use_batch_norm=True, is_training=self.is_training,
                                      use_leak=True)
                dconv4 = model.deconv(dconv3, [3, 3, 1, dims], [self.opts.batch_size, 28, 28, 1], 2, "output", None,
                                      initializer=tf.truncated_normal_initializer(stddev=0.2), \
                                      bias_constant=0.0, reuse=reuse)

                # full1_reshape = tf.reshape(full1, [-1, 7, 7, dims*2])
                # dconv2 = model.deconv(full1_reshape, [3,3,dims,dims*2], [self.opts.batch_size, 14, 14, dims], 2, "dconv2", activation=tf.nn.relu, initializer=tf.truncated_normal_initializer(stddev=0.2),
                # 					  bias_constant=0.0, reuse=reuse, use_batch_norm=True, is_training=self.is_training, use_leak=True)
                # dconv3 = model.deconv(dconv2, [3,3,1,dims], [self.opts.batch_size, 28, 28, 1], 2, "dconv3", activation=None, initializer=tf.truncated_normal_initializer(stddev=0.2),\
                # 					  bias_constant=0.0, reuse=reuse, use_batch_norm=False, is_training=self.is_training, use_leak=True)
                return tf.nn.sigmoid(dconv4)

    def loss(self):
        with tf.variable_scope("loss"):
            if not self.opts.use_labels:
                true_prob = tf.nn.sigmoid(self.true_logit)
                fake_prob = tf.nn.sigmoid(self.fake_logit)
            with tf.variable_scope("D_loss"):
                if self.opts.label_len > 1:
                    d_true_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.true_logit,
                                                                          dim=1)
                    d_fake_loss = tf.nn.softmax_cross_entropy_with_logits(labels=1 - self.labels,
                                                                          logits=self.fake_logit, dim=1)
                    d_loss = d_true_loss + d_fake_loss
                else:
                    d_loss = tf.reduce_mean(-tf.log(true_prob) - tf.log(1 - fake_prob))
            with tf.variable_scope("G_loss"):
                if self.opts.label_len > 1:
                    g_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.fake_logit, dim=1)
                else:
                    g_loss = tf.reduce_mean(-tf.log(fake_prob))
            return tf.reduce_mean(d_loss), tf.reduce_mean(g_loss)

    def train(self):
        code = np.random.uniform(low=-1.0, high=1.0, size=[self.opts.batch_size, self.opts.code_len]).astype(np.float32)
        utils = Dataset(self.opts)
        D_lr = self.opts.D_base_lr
        G_lr = self.opts.G_base_lr
        self.sess.run(self.init)
        for iteration in range(1, self.opts.MAX_iterations):
            batch_num = 0
            for batch_begin, batch_end in zip(range(0, self.opts.train_size, self.opts.batch_size), \
                                              range(self.opts.batch_size, self.opts.train_size, self.opts.batch_size)):
                begin_time = time.time()
                if self.opts.use_labels:
                    batch_imgs, batch_labels = utils.load_batch(batch_begin, batch_end)
                else:
                    batch_imgs = utils.load_batch(batch_begin, batch_end)
                noise = np.random.uniform(low=-1.0, high=1.0, size=[self.opts.batch_size, self.opts.code_len]).astype(
                    np.float32)

                # Real data
                if self.opts.use_labels:
                    feed_dict = {self.images: batch_imgs, self.D_lr: D_lr, self.G_lr: G_lr, self.code: noise,
                                 self.labels: batch_labels}
                else:
                    feed_dict = {self.images: batch_imgs, self.D_lr: D_lr, self.G_lr: G_lr, self.code: noise}
                _, D_loss = self.sess.run([self.D_optimizer, self.d_loss], feed_dict=feed_dict)

                # Fake data
                if self.opts.use_labels:
                    feed_dict = {self.images: batch_imgs, self.D_lr: D_lr, self.G_lr: G_lr, self.code: noise,
                                 self.labels: batch_labels}
                else:
                    feed_dict = {self.images: batch_imgs, self.D_lr: D_lr, self.G_lr: G_lr, self.code: noise}
                _, G_loss, summary = self.sess.run([self.G_optimizer, self.g_loss, self.summaries], feed_dict=feed_dict)

                batch_num += 1
                self.writer.add_summary(summary, iteration * (self.opts.train_size / self.opts.batch_size) + batch_num)
                if batch_num % self.opts.display == 0:
                    rem_time = (time.time() - begin_time) * (self.opts.MAX_iterations - iteration) * (
                        self.opts.train_size / self.opts.batch_size)
                    log = '-' * 20
                    log += '\nIteration: {}/{}|'.format(iteration, self.opts.MAX_iterations)
                    log += ' Batch Number: {}/{}|'.format(batch_num, self.opts.train_size / self.opts.batch_size)
                    log += ' Batch Time: {}\n'.format(time.time() - begin_time)
                    # log += ' Remaining Time: {:0>8}\n'.format(datetime.timedelta(seconds=rem_time))
                    log += ' D_lr: {} D_loss: {}\n'.format(D_lr, D_loss)
                    log += ' G_lr: {} G_loss: {}\n'.format(G_lr, G_loss)
                    print(log)
                if iteration % self.opts.lr_decay == 0 and batch_num == 1:
                    D_lr *= self.opts.lr_decay_factor
                    G_lr *= self.opts.lr_decay_factor
                if iteration % self.opts.ckpt_frq == 0 and batch_num == 1:
                    self.saver.save(self.sess,
                                    self.opts.root_dir + self.opts.ckpt_dir + "{}_{}_{}_{}".format(iteration, D_lr,
                                                                                                   G_lr,
                                                                                                   D_loss + G_loss))
                if iteration % self.opts.generate_frq == 0 and batch_num == 1:
                    feed_dict = {self.code: code}
                    self.is_training = False
                    imgs = self.sess.run(self.generated_imgs, feed_dict=feed_dict)
                    if self.opts.dataset == "CIFAR":
                        imgs = np.reshape(imgs, (self.opts.test_size, 3, 32, 32)).transpose(0, 2, 3, 1)
                    else:
                        imgs = np.reshape(imgs, (self.opts.test_size, 28, 28))
                    utils.save_batch_images(imgs, [self.opts.grid_h, self.opts.grid_w], str(iteration) + ".jpg", True)
                    self.is_training = True


class VAE(object):
    """
    Variatinoal Autoencoder
    """

    def __init__(self, opts, is_training=True):
        self.opts = opts
        self.h = opts.IMG_H
        self.w = opts.IMG_W
        self.c = opts.CHANNELS

        self.is_training = self.opts.TRAIN

    def resnet_out(self, inputs):
        print("get hps")
        self.hps = resnet.HParams(batch_size=self.opts.BATCH_SIZE,
                                  num_classes=self.opts.ENCODE_VECTOR_SIZE,
                                  min_lrn_rate=self.opts.MIN_LR,
                                  lrn_rate=self.opts.BASE_LR,
                                  num_residual_units=self.opts.NUM_RESIDUAL_UNITS,
                                  use_bottleneck=False,
                                  weight_decay_rate=0.0002,
                                  relu_leakiness=0.1,
                                  optimizer='mom')
        print("Done")
        if self.is_training:
            self.mode = "train"
        else:
            self.mode = "eval"

        print("creat resnet model")
        resmodel = resnet.ResNet(self.hps, inputs, mode=self.mode)
        print("Done")

        print("build forward graph")
        resmodel.build_graph()
        print("Done")

        conv_out = resmodel.conv_out
        print("conv_out shape = ", conv_out.get_shape())
        return conv_out

    def encoder(self, inputs):
        print("get conv out")
        conv_out = self.resnet_out(inputs)
        print("Done")
        self.conv_flat_len = int(np.prod(conv_out.get_shape()[1:]))

        conv_out_flat = tf.reshape(conv_out, [-1, self.conv_flat_len])

        #self.resnet_out = conv_out_flat

        print("resnet out flat shape = ", conv_out_flat.get_shape())
        mean = model.fully_connected(conv_out_flat,
                                     self.opts.ENCODE_VECTOR_SIZE,
                                     self.is_training,
                                     None, "full5_mean",
                                     use_leak=True,
                                     bias_constant=0.01)  # 40

        print('mean shape = {mean_shape}'.format(mean_shape=mean.get_shape()))

        stds = model.fully_connected(conv_out_flat,
                                     self.opts.ENCODE_VECTOR_SIZE,
                                     self.is_training,
                                     None, "full5_stds",
                                     use_leak=True,
                                     bias_constant=0.01)  # 40

        print('stds shape = {stds_shape}'.format(stds_shape=stds.get_shape()))

        return mean, stds

    def dconv_resnet_out(self, z):
        dims = self.opts.DIMS

        print("get hps")
        self.hps_dconv = resnet.HParams(batch_size=self.opts.BATCH_SIZE,
                                        num_classes=self.opts.ENCODE_VECTOR_SIZE,
                                        min_lrn_rate=self.opts.MIN_LR,
                                        lrn_rate=self.opts.BASE_LR,
                                        num_residual_units=self.opts.NUM_RESIDUAL_UNITS,
                                        use_bottleneck=False,
                                        weight_decay_rate=0.0002,
                                        relu_leakiness=0.1,
                                        optimizer='mom')
        print("Done")

        if self.is_training:
            self.mode = "train"

        else:
            self.mode = "eval"

        print("get dconv model")
        dconv_resmodel = dconv_resnet.ResNetDeconv(self.hps_dconv, z, mode=self.mode, opts=self.opts, dims=dims)
        print("Done")

        print("get dconv graph")
        dconv_resmodel.build_graph()
        print("Done")

        dconv_out = dconv_resmodel.dconv_out

        return dconv_out

    def decoder(self, z):
        """
        Generate images from the `latent vector`

        def deconv(input, kernel, output_shape, stride=1, name=None,
	    activation=None, use_batch_norm=False, is_training=False,
	    reuse=False, initializer=tf.contrib.layers.xavier_initializer(),
	    bias_constant=0.0, use_leak=False, alpha=0.0):


        """

        dims = self.opts.DIMS

        with tf.variable_scope("decoder"):
            full1 = model.fully_connected(z, self.conv_flat_len, self.is_training, tf.nn.relu, "full1",
                                          use_leak=True, alpha=0.2)  # 4x4x256/ 4*4*512
            print('full1 shape = {full1}'.format(full1=full1.get_shape()))

            dconv_in = tf.reshape(full1, [-1, 5, 5, dims * (32)])

            print('dconv in = {dconvin_shape}'.format(dconvin_shape=dconv_in.get_shape()))
            dconv_out = self.dconv_resnet_out(dconv_in)

            print("dconv_out shape = ", dconv_out.get_shape())
            
            '''output_1 = model.deconv(dconv_out, [3, 3, 8, 16], [self.opts.BATCH_SIZE, self.h, self.w, 8],
                                  1,
                                  "output_1", initializer=tf.truncated_normal_initializer(stddev=0.02), use_leak=True,
                                  alpha=0.2)  
            print('output shape = {dconv_output}'.format(dconv_output=output_1.get_shape()))'''

            output = model.deconv(dconv_out, [3, 3, self.c, 8], [self.opts.BATCH_SIZE, self.h, self.w, self.c],
                                  1,
                                  "output", initializer=tf.truncated_normal_initializer(stddev=0.02), use_leak=True,
                                  alpha=0.2)  # 32x32x3/ 128*128*3
            print('output shape = {dconv_output}'.format(dconv_output=output.get_shape()))
            
            probs = tf.nn.sigmoid(output)
            print('probs shape = {p}'.format(p=probs.get_shape()))

        return probs

    def loss(self, mean, std, images, generated_imgs):

        img_flat = tf.reshape(images, [-1, self.h * self.w * self.c])
        gen_flat = tf.reshape(generated_imgs, [-1, self.h * self.w * self.c])
        encoder_loss = 0.5 * tf.reduce_sum(
            tf.square(mean) + tf.square(std) - tf.log(tf.clip_by_value(tf.square(std), 1e-10, 1)) - 1.,
            1)

        decoder_loss = -tf.reduce_sum(
            img_flat * tf.log(tf.clip_by_value(1e-8 + gen_flat, 1e-10, 1)) + (1 - img_flat) * tf.log(
                tf.clip_by_value(1e-8 + 1 - gen_flat, 1e-10, 1)), 1)

        el = self.opts.D_LAMDA * tf.reduce_mean(encoder_loss)
        dl = self.opts.G_LAMDA * tf.reduce_mean(decoder_loss)

        print(el.get_shape(), dl.get_shape())
        return el, dl


    def get_loss(self, x, scope, reuse_variables=None):
        # 沿用5.5节中定义的函数来计算神经网络的前向传播结果。
        print("input batch shape =", x.get_shape())      
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
            print("get mean and std")
            mean, std = self.encoder(x)
            print("Done")

            unit_gauss = tf.random_normal([self.opts.BATCH_SIZE, self.opts.ENCODE_VECTOR_SIZE])
            print("get z")
            z = mean + std * unit_gauss
            print("Done")

            print("get imgs")
            generated_imgs = self.decoder(z)
            print("Done")

            print("get l1 and l2")
            l1, l2 = self.loss(mean, std, x, generated_imgs)
            print("Done")
            # Assemble all of the losses for the current tower only.
            tf.add_to_collection('losses', l1)
            tf.add_to_collection('losses', l2)

            print("losses")
            losses = tf.get_collection('losses', scope)
            print("Done,losses shaep = ", losses)

            # Calculate the total loss for the current tower.
            print("total loss")
            total_loss = tf.add_n(losses, name='total_loss')
            print("Done,total loss = ", total_loss)
            # Attach a scalar summary to all individual losses and the total loss; do the
            # same for the averaged version of the losses.
            for l in losses + [total_loss]:
                # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
                # session. This helps the clarity of presentation on tensorboard.
                loss_name = re.sub('_[0-9]*/', '', l.op.name)
                tf.summary.scalar(loss_name, l)

            return l1, l2, total_loss, generated_imgs, mean

    def average_grident(self, tower_grads):
        average_grads = []
        for val_and_grad in zip(*tower_grads):
            grads = []
            for g, _ in val_and_grad:
                grad = tf.expand_dims(g, 0)
                grads.append(grad)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = val_and_grad[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def main(self):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False)
            num_batches_per_epoch = self.opts.TRAIN_SIZE // self.opts.BATCH_SIZE

            lr = tf.train.exponential_decay(self.opts.BASE_LR,
                                            global_step,
                                            self.opts.DECAY_AFTER_GS,
                                            self.opts.LR_DECAY,
                                            staircase=True)

            opt = tf.train.AdamOptimizer(lr)
            utils_set = utils.Dataset(self.opts)
            image_batch = utils_set.get_batch()
            print("prepare batch")

            tower_grad = []

            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.opts.NUM_GPU):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('GPU_%d' % i) as scope:
                            print("gpu_%d" % i)
                            # image_batch = batch_queue.dequeue()
                            encode_loss, decode_loss, cur_loss, generated_images, generated_code = self.get_loss(image_batch, scope=scope)
                            print("gpu_%d loss done" % i)
                            tf.get_variable_scope().reuse_variables()
                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                            grads = opt.compute_gradients(cur_loss)
                            tower_grad.append(grads)
                    summaries.append(tf.summary.scalar('loss', cur_loss))
                    summaries.append(tf.summary.scalar('encode loss', encode_loss))
                    summaries.append(tf.summary.scalar('decode loss', decode_loss))
                    summaries.append(tf.summary.scalar('learning_rate', lr))

            grads = self.average_grident(tower_grad)

            print("1 done")
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram('%s' % var.op.name, grad)

            apply_grident_op = opt.apply_gradients(grads, global_step=global_step)
            print("2 done")

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            print("3 done")

            variable_averages = tf.train.ExponentialMovingAverage(0.9, global_step)
            variable_average_op = variable_averages.apply(tf.trainable_variables())

            train_op = tf.group(apply_grident_op, variable_average_op)

            saver = tf.train.Saver(tf.global_variables(),max_to_keep=self.opts.MAX_TO_KEEP)
            summary_op = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.

            restore = self.opts.RESTORE
            restore_model = self.opts.RESTORE_MODEL

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.opts.GPU_PERCENTAGE)

            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=True,
                    gpu_options=gpu_options)) as self.sess:

                if restore == False:
                    self.sess.run(init)
                else:
                    saver.restore(self.sess, restore_model)

                # Start the queue runners.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord,sess=self.sess)

                summary_writer = tf.summary.FileWriter(os.path.join(self.opts.ROOT_DIR, self.opts.SUMMARY_DIR),
                                                       self.sess.graph)
                
                

                for iteration in range(1, self.opts.MAX_ITERATIONS):
                    
                    if iteration >= self.opts.MAX_ITERATIONS-1:
                        print("cord stop")
                        coord.request_stop()
                    
                    try:
            
                        if not coord.should_stop():
                            
                            for batch_num in range(num_batches_per_epoch):
                                
        
                                start_time = time.time()
                                
                                #_, lr_value, el_value, dl_value, loss_value, generated_images_value= self.sess.run([train_op, lr, encode_loss, decode_loss, cur_loss, generated_images])
                                _, lr_value, el_value, dl_value, loss_value, generated_images_value, generated_code_value, image_batch_value= self.sess.run([train_op, lr, encode_loss, decode_loss, cur_loss, generated_images, generated_code, image_batch])
                

                                duration = time.time() - start_time
        
                                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        
                                if iteration % self.opts.LOG_DECAY == 0 and batch_num == 1:
                                    summary = self.sess.run(summary_op)
                                    summary_writer.add_summary(summary, iteration * (num_batches_per_epoch + batch_num))
        
                                if batch_num % self.opts.DISPLAY == 0:
                                    
                                    log = '-' * 20
                                    log += '\nIteration: {}/{}|'.format(iteration, self.opts.MAX_ITERATIONS)
                                    log += ' Batch Number: {}/{}|'.format(batch_num,
                                                                          self.opts.TRAIN_SIZE / self.opts.BATCH_SIZE)
                                    log += ' Batch Time: {}\n'.format(duration)
                                    log += ' Learning Rate: {}\n'.format(lr_value)
                                    log += ' Encoder Loss: {}\n'.format(el_value)
                                    log += ' Decoder Loss: {}\n'.format(dl_value)
                                    log += ' code example: {}\n'.format(generated_code_value)
                                    log += ' image example: {}\n'.format(image_batch_value)
                                    
                                    print(log)
                                # if iteration % self.opts.lr_decay == 0 and batch_num == 1:
                                #     lr *= self.opts.lr_decay_factor
                                if iteration % self.opts.CKPT_FRQ == 0 and batch_num == 1:
                                    print("save ckpt ... ...")
                                    saver.save(self.sess,
                                               os.path.join(self.opts.ROOT_DIR, self.opts.CKPT_DIR,
                                                            "{}".format(iteration)))
                                if iteration % self.opts.GENERATE_FRQ == 0 and batch_num == 1:
                                    print("generate images ... ...")
                                    #generate_imgs = tf.convert_to_tensor(utils_set.test_images)
                                    #generated_imgs = self.decoder(self.encoder(generate_imgs))
                                    #imgs = self.sess.run(generated_imgs)
                                    #imgs = np.reshape(imgs, (self.opts.TEST_SIZE, 3, 128, 128)).transpose(0, 2, 3, 1)
        
                                    #tf.summary.image('Generated image', imgs[0])
                                    print(generated_images_value.shape)
                                    utils_set.save_batch_images(generated_images_value, [self.opts.GRID_H, self.opts.GRID_W], str(iteration) + ".jpg",
                                                            True)

                    except tf.errors.OutOfRangeError:
                        
                        print ('Done training -- epoch limit reached')
                        break
                print("End training")
                # When done, ask the threads to stop. 请求该线程停止
                coord.request_stop()
                # And wait for them to actually do it. 等待被指定的线程终止
                coord.join(threads)
                
    def test(self, image):
        latest_ckpt = tf.train.latest_checkpoint(self.opts.CKPT_DIR)
        #latest_ckpt = '/home/wz/LiYi/VAE_GAN/Generative-Models2/ckpt/vae/300'
        latest_ckpt = self.opts.RESTORE_MODEL
        image = tf.div(image,255)
        mean, std = self.encoder(image)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, latest_ckpt)
            mean_out, std_out = sess.run([mean, std])
            encode_out_value = mean_out

        return encode_out_value

    def cluster(self, image_dir):
        
        K=3
        
        print(__doc__)
        from time import time

        import numpy as np
        import math
        import matplotlib.pyplot as plt
        from matplotlib import offsetbox
        from sklearn import manifold
        from PIL import Image
        
        def plot_embedding(X, title=None):
            x_min, x_max = np.min(X, 0), np.max(X, 0)
            X = (X - x_min) / (x_max - x_min)

            plt.figure()
            ax = plt.subplot(111)

            if hasattr(offsetbox, 'AnnotationBbox'):
                # only print thumbnails with matplotlib > 1.0
                shown_images = np.array([[1., 1.]])  # just something big
                for i in range(digits["data"].shape[0]):
                    dist = np.sum((X[i] - shown_images) ** 2, 1)
                    if np.min(dist) < 4e-3:
                        # don't show points that are too close
                        continue
                    shown_images = np.r_[shown_images, [X[i]]]
                    imagebox = offsetbox.AnnotationBbox(
                        offsetbox.OffsetImage(digits["images_resize"][i]),
                        X[i])
                    ax.add_artist(imagebox)
            plt.xticks([]), plt.yticks([])
            if title is not None:
                plt.title(title)
                
        def plot_embedding2(X, title=None):
            
            cordinate = []

            count_dict = dict()
            
            #以list存储所有result，result=[当前该类个数，类别名]
            for elem in X:
                print(elem)
                cur = [count_dict.setdefault(elem,0),elem]
                cordinate.append(np.array(cur))
                count_dict[elem] += 1
                
                
            plt.figure()
            if title is not None:
                plt.title(title)
            class_num = len(count_dict.keys())
            class_num = K
            
            col = math.floor(math.sqrt(class_num))
            row = math.ceil(class_num / col)
            
            print('classnum = ', class_num, 'col = ',col, 'row = ', row)
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            if hasattr(offsetbox, 'AnnotationBbox'):
                #digits_test = digits["images_resize"][:len(X)]
                digits_test = digits["images_resize"]
                #for i in range(digits["data"].shape[0]):

                #for i in range(digits_test.shape[0]):
                for i in range(len(digits_test)):
                    cur_cordinate = np.random.uniform(low=0,high=1,size=(2))
                    imagebox = offsetbox.AnnotationBbox(
                        offsetbox.OffsetImage(digits["images_resize"][i]),
                        cur_cordinate)
  
                    plt.subplot(row,col,cordinate[i][1]+1).add_artist(imagebox)
                
            for i in range(class_num):
                plt.subplot(row,col,i+1).set_title(str(i)+' PASI leavel')
                
            plt.xticks([]), plt.yticks([])
            plt.tight_layout()
            
            
        from sklearn.externals import joblib
        if self.opts.TRAIN_GMM:
            
            from sklearn.mixture import BayesianGaussianMixture as BayesGMM
            from sklearn.decomposition import PCA
            
            #latest_ckpt = tf.train.latest_checkpoint(self.opts.CKPT_DIR)
            latest_ckpt = self.opts.RESTORE_MODEL
            
            image = tf.placeholder(tf.float32, shape=(None, self.opts.IMG_H, self.opts.IMG_W, 3))
            image = tf.div(image,255)
            mean, std = self.encoder(image)
            saver = tf.train.Saver()
            
            
            with tf.Session() as sess:
                saver.restore(sess, latest_ckpt)
            
                digits = dict()
                digits["data"] = []
                print("begin to load data")
                file_list = os.listdir(image_dir)
                imgs_file = [os.path.join(image_dir, filename) for filename in file_list]
                
                
                batchSize = 256
                batchNum = math.floor(len(imgs_file) / batchSize)
                
                
                t0 = time()
                for i in range(batchNum):
                    
                    digits["images"] = []
                    print('batch', i)
        
                    for file in imgs_file[i*batchSize:(i+1)*batchSize]:
                        img = Image.open(file)
                        img = np.array(img)
                        digits['images'].append(img)
                    digits['images'] = np.array(digits['images'])
                    digits["images"] = digits["images"].astype('float')
                    
                    
                    print("begin to encode images")
                    mean_out, std_out = sess.run([mean, std],feed_dict={image:digits['images']})
    
                    #digits["data"] = mean_out
                    digits["data"].extend(mean_out)
            
                X = digits["data"]
                    
                    #print(X)
                    
                    
                print("Computing GMM embedding")
                # Fit a Dirichlet process Gaussian mixture using five components
                
                '''
                print("demension reducing ... ...")
                pca_X = PCA(n_components=K**2).fit_transform(X)
                print("demension reduced ... ...")
                '''
        
                #pca_X_train = pca_X[2000:]

                dpgmm = BayesGMM(n_components=K,
                                 n_init=10,
                                 covariance_type='full',
                                 init_params='kmeans',
                                 warm_start=True,
                                 max_iter=500,
                                 verbose=2,
                                 weight_concentration_prior_type='dirichlet_distribution'
                                 ).fit(X)
                    
                '''    
                digits["images"] = []
                print('batch', batchNum)
    
                for file in imgs_file[batchNum*batchSize:]:
                    img = Image.open(file)
                    img = np.array(img)
                    digits['images'].append(img)
                digits['images'] = np.array(digits['images'])
                digits["images"] = digits["images"].astype('float')
                
                
                print("begin to encode images")
                mean_out, std_out = sess.run([mean, std],feed_dict={image:digits['images']})

                digits["data"] = mean_out
        
                X = digits["data"] 
                print("Computing GMM embedding")
                   
                K = 5
                # Fit a Dirichlet process Gaussian mixture using five components
      
                pca_X = X
                dpgmm = BayesGMM(n_components=K,
                                 n_init=50,
                                 covariance_type='full',
                                 init_params='kmeans',
                                 warm_start=True,
                                 weight_concentration_prior_type='dirichlet_distribution'
                                 ).fit(pca_X)'''
            
            print('saving GMM... ...')
            joblib.dump(dpgmm, "/home/wz/HD1/LiYi/VAE_GAN/GMM/dpgmm.m")
            print('Done')
            
            
        else:
            from sklearn.mixture import BayesianGaussianMixture as BayesGMM
            from sklearn.decomposition import PCA
            
            #latest_ckpt = tf.train.latest_checkpoint(self.opts.CKPT_DIR)
            latest_ckpt = self.opts.RESTORE_MODEL
            X_predict = []
            
            image = tf.placeholder(tf.float32, shape=(None, self.opts.IMG_H, self.opts.IMG_W, 3))
            image = tf.div(image,255)
            mean, std = self.encoder(image)
            saver = tf.train.Saver()
            
            
            with tf.Session() as sess:
                saver.restore(sess, latest_ckpt)
            
                digits = dict()
                digits["images_resize"] = []
                
                print("begin to load data")
                file_list = os.listdir(image_dir)
                imgs_file = [os.path.join(image_dir, filename) for filename in file_list][:1000]
                
                
                batchSize = 256
                batchNum = math.floor(len(imgs_file) / batchSize)
                
                dpgmm = joblib.load("/home/wz/HD1/LiYi/VAE_GAN/GMM/dpgmm.m")
                
                
                t0 = time()
                for i in range(batchNum):
                    
                    digits["images"] = []
                    print('batch', i)
        
                    for file in imgs_file[i*batchSize:(i+1)*batchSize]:
                        img = Image.open(file)                      
                        digits["images_resize"].append(img.resize((32,32)))
                        img = np.array(img)
                        digits['images'].append(img)
                    digits['images'] = np.array(digits['images'])
                    digits["images"] = digits["images"].astype('float')
                    
                    
                    print("begin to encode images")
                    mean_out, std_out = sess.run([mean, std],feed_dict={image:digits['images']})
    
                    digits["data"] = mean_out
            
                    X = digits["data"]
                    print('predicting ... ...')
            
                    X_gmm = dpgmm.predict(X)
                    X_predict.extend(X_gmm)
                    
                    
                digits["images"] = []
                print('batch', batchNum)
                for file in imgs_file[batchNum*batchSize:]:
                    img = Image.open(file)                      
                    digits["images_resize"].append(img.resize((32,32)))
                    img = np.array(img)
                    digits['images'].append(img)
                digits['images'] = np.array(digits['images'])
                digits["images"] = digits["images"].astype('float')
                
                #digits['images_resize'] = np.array(digits['images_resize'])
                print("begin to encode images")
                mean_out, std_out = sess.run([mean, std],feed_dict={image:digits['images']})
    
                digits["data"] = mean_out
        
                X = digits["data"]
                print('predicting ... ...')
        
                X_gmm = dpgmm.predict(X)
                X_predict.extend(X_gmm)
         
            print('GMM means = ',dpgmm.means_)
            print('GMM weights = ',dpgmm.weights_)
            print('GMM tied covariance = ',dpgmm.covariances_)
            print('GMM params = ',dpgmm.get_params)
            
            print('ploting ... ...')  
            plot_embedding2(X_predict,
                           "GMM embedding (time %.2fs)" %
                           (time() - t0))
    
            plt.savefig(os.path.join(self.opts.ROOT_DIR, os.path.join(self.opts.SAMPLE_DIR,"GMM.jpg")))
            plt.show()
        
        
        
        # t-SNE embedding of dataset
#         print("Computing t-SNE embedding")
#         tsne = manifold.TSNE(n_components=2,learning_rate=100)
#         t0 = time()
#         X_tsne = tsne.fit_transform(X)
        

#         print(X_tsne.shape)
#         plot_embedding(X_tsne,
#                        "t-SNE embedding (time %.2fs)" %
#                        (time() - t0))

#         plt.savefig(os.path.join(self.opts.ROOT_DIR, os.path.join(self.opts.SAMPLE_DIR,"t-sne.jpg")))
#         plt.show()
        
        
        





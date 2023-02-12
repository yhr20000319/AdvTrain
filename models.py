import scipy as sp

from layers import *
from metrics import *
import numpy as np
import tensorflow as tf2
import torch

from utils import preprocess_adj, load_data

tf = tf2.compat.v1
tf.disable_v2_behavior()

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None

        self.outputs = None
        self.outputs2 = None
        self.h1 = None
        self.h2 = None

        self.adv_rate = 0
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])  # layer输入上一次的隐藏层
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        # self.outputs, self.outputs2, self.h1, self.h2 = self.outputs[0], self.outputs[1], self.outputs[2], self.outputs[3]
        self.outputs, self.outputs2 = self.outputs[0], self.outputs[1]
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._attack_loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def build_FPAT(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])  # layer输入上一次的隐藏层
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        self.outputs, self.h1, self.h2 = self.outputs[0], self.outputs[1], self.outputs[2]
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._FPAT_loss()
        self._attack_loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def build_trades(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])  # layer输入上一次的隐藏层
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        # self.outputs, self.outputs2, self.h1, self.h2 = self.outputs[0], self.outputs[1], self.outputs[2], self.outputs[3]
        self.outputs, self.outputs2 = self.outputs[0], self.outputs[1]
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._trades_loss()
        self._attack_loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _FPAT_loss(self):
        raise NotImplementedError

    def _trades_loss(self):
        raise NotImplementedError

    def _attack_loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None, path=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        if not path:
            save_path = saver.save(sess, "tmp1/%s.ckpt" % self.name)
        else:
            save_path = saver.save(sess, path)
        print("Model saved in file: %s" % save_path)

    def load_original(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp1/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

    def load(self, path, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = tf.train.latest_checkpoint(path)
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        '''
        首先作者用tf.nn.softmax_cross_entropy_with_logits求出了每一行训练样本的softmax交叉熵，
        具体是直接把第二层卷积的结果（140,7）直接softmax之后，与y计算交叉熵，
        然后屏蔽掉值中非训练集的y值，避免这些结果算进loss里面去，
        作者将placeholders['labels_mask'])（实际上是train_mask）
        从[True,True...False]转化为[1,1,1,...0]（前140个元素是1，属于训练集），
        mask /= tf.reduce_mean(mask)目的是在return的时候对loss的均值开始包括了其他遮蔽的值，
        因此此时在分子做扩大补充，那mask就是[19.34,19.34,19.34...0]即遮蔽掉的为0，没遮蔽的全部除以140/2708，
        最后每一行的交叉熵和每一行对应的mask值相乘得到最终的loss

        '''
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, attack=None, trades=False, FPAT=False, adv_rate=0, **kwargs):
        super(GCN, self).__init__(**kwargs)
        print('attack method:', attack)
        # if attack is False, placeholders['support'] feeds in normalized pre-processed adjacent matrix, 
        # if attack is True, placeholders['adj'] feeds in raw adjacent matrix and placeholdder['s'] feeds in attack placeholders
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        print("labels: ", placeholders['labels'].get_shape().as_list())
        self.placeholders = placeholders
        lmd = placeholders['lmd']
        self.attack = attack
        self.trades = trades
        self.FPAT = FPAT
        self.adv_rate = adv_rate

        if self.attack:
            mu = placeholders['mu']  # 对偶变量

            # the length of A list, in fact, self.num_support is always 1
            self.num_supports = len(placeholders['adj'])
            # original adjacent matrix A
            self.A = placeholders['adj']
            # np.triu,1，获取矩阵不含对角线的上三角部分
            self.mask = [tf.constant(np.triu(np.ones([self.A[0].get_shape()[0].value] * 2, dtype=np.float32), 1))]

            #  C 代表可以添加的边
            self.C = [1 - 2 * self.A[i] - tf.eye(self.A[i].get_shape().as_list()[0], self.A[i].get_shape().as_list()[1])
                      for i in range(self.num_supports)]
            # placeholder for adding edges
            self.upper_S_0 = placeholders['s']  # 大小n*n 每次更新
            self.SS = placeholders['ss']  # 大小为n 每次更新
            # a strict upper triangular matrix to ensure only N(N-1)/2 trainable variables
            # here use matrix_band_part to ensure a stricly upper triangular matrix  n（n-1）/2因为是上三角矩阵不含对角的元素个数
            self.upper_S_real = [
                tf.matrix_band_part(self.upper_S_0[i], 0, -1) - tf.matrix_band_part(self.upper_S_0[i], 0, 0) for i in
                range(self.num_supports)]
            # modified_A is the new adjacent matrix
            self.upper_S_real2 = [self.upper_S_real[i] + tf.transpose(self.upper_S_real[i]) for i in
                                  range(self.num_supports)]
            self.modified_A = [self.A[i] + tf.multiply(self.upper_S_real2[i], self.C[i]) for i in
                               range(self.num_supports)]
            """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
            self.hat_A = [tf.cast(self.modified_A[i] + tf.eye(self.modified_A[i].get_shape().as_list()[0],
                                                              self.modified_A[i].get_shape().as_list()[1]),
                                  dtype='float32') for i in range(self.num_supports)]
            # hat_A为[hat_A[0]] ,hat_A[0]为modi_A+I n*n
            # get degree by row sum
            self.rowsum = tf.reduce_sum(self.hat_A[0], axis=1)  # 即n维向量，代表n个节点的度，D
            self.d_sqrt = tf.sqrt(self.rowsum)  # 平方根，即D1/2
            self.d_sqrt_inv = tf.math.reciprocal(self.d_sqrt)  # 计算倒数，即D-1/2
            self.support_real = tf.multiply(tf.transpose(tf.multiply(self.hat_A[0], self.d_sqrt_inv)), self.d_sqrt_inv)
            # D-1/2*A*D-1/2
            # this self.support is a list of \tilde{A} in the paper
            # replace the 'support' in the placeholders dictionary

            self.placeholders['support'] = [self.support_real]

            self.rowsum = tf.reduce_sum(self.A[0], axis=1)
            self.d_sqrt = tf.sqrt(self.rowsum)  # 平方根，即D1/2
            self.d_sqrt_inv = tf.math.reciprocal(self.d_sqrt)  # 计算倒数，即D-1/2
            self.hat_A = [tf.cast(self.A[i] + tf.eye(self.A[i].get_shape().as_list()[0],
                                                     self.A[i].get_shape().as_list()[1]),
                                  dtype='float32') for i in range(self.num_supports)]
            self.ori_support = tf.multiply(tf.transpose(tf.multiply(self.hat_A[0], self.d_sqrt_inv)), self.d_sqrt_inv)
            self.placeholders['ori_support'] = [self.ori_support]
            # 此处可将对抗样本 A' 修改为 λ origin_A + (1-λ) A'
            # self.placeholders['support'] = [0.5 * self.support_real + 0.5 * self.ori_support]

            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            if self.trades:
                self.build_trades()
            elif self.FPAT:
                self.build_FPAT()
            else:
                self.build()

            # proximal gradient algorithm
            if self.attack == 'PGD':
                self.Sgrad = tf.gradients(self.attack_loss,
                                          self.upper_S_real[0])  # ys, xs，实现ys对xs求导，求导返回值是一个list，list的长度等于len(xs)
                self.a = self.upper_S_real[0] + mu * self.Sgrad * lmd * self.mask  # 更新，之前的扰动矩阵加上新的梯度步长和μ
            elif self.attack == 'SPEC':
                # 为ori_support计算特征值，特征向量
                eigenvalues, eigenvectors = tf.self_adjoint_eig(self.ori_support)  # e是n维，v是n*n维 L = η Λ η−1 = η Λ ηT
                # eigenvalues, eigenvectors = tf.self_adjoint_eig(self.support_real)
                # 定义待优化目标函数 ||λ||2
                target = tf.subtract(self.SS, eigenvalues)
                self.loss1 = tf.norm(target)
                self.Sgrad = tf.gradients(self.loss1, self.SS)  # 最大化特征值变化方向
                # 用特征值变化还原上三角变化矩阵
                self.S2 = mu * self.Sgrad * lmd * 100
                delta_S = tf.matmul(tf.multiply(eigenvectors, self.S2), tf.transpose(eigenvectors))
                self.a = self.upper_S_real[0] + delta_S * self.mask
            elif self.attack == 'CW':
                label = placeholders['labels']
                real = tf.reduce_sum(label * self.outputs, 1)  # output n*7
                label_mask_expand = placeholders['label_mask_expand']
                other = tf.reduce_max((1 - label) * label_mask_expand * self.outputs - label * 10000, 1)
                self.loss1 = tf.maximum(0.0, (real - other + 50) * label_mask_expand[:, 0])
                self.loss2 = tf.reduce_sum(self.loss1)
                self.Sgrad = tf.gradients(self.loss2, self.upper_S_real[0])
                self.a = self.upper_S_real[0] - mu * self.Sgrad * lmd * self.mask
            elif self.attack == 'minmax':
                self.w = placeholders['w']
                label = placeholders['labels']
                self.real = tf.reduce_sum(label * self.outputs, 1)
                label_mask_expand = placeholders['label_mask_expand']
                self.other = tf.reduce_max((1 - label) * label_mask_expand * self.outputs - label * 10000, 1)
                self.loss1 = self.w * tf.maximum(0.0, self.real - self.other + 0.)
                self.loss2 = tf.reduce_sum(self.loss1)
                self.Sgrad = tf.gradients(self.loss2, self.upper_S_real[0])
                self.a = self.upper_S_real[0] - mu * self.Sgrad * self.mask
            else:
                raise NotImplementedError

        else:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            self.build()

    def _attack_loss(self):
        # Cross entropy error
        self.attack_loss = masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                        self.placeholders['labels_mask'])

    def _loss(self):
        # Weight decay loss 加了一个惩罚项，权重衰减
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _FPAT_loss(self):
        # Weight decay loss 加了一个惩罚项，权重衰减
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

        # add 隐藏层loss
        self.loss += 1 * masked_softmax_KL(self.h1, self.h2,
                                      self.placeholders['labels_mask'])

    def _trades_loss(self):
        # Weight decay loss 加了一个惩罚项，权重衰减
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        # adv_loss
        self.loss += self.adv_rate * masked_softmax_KL(self.outputs, self.outputs2,
                                                       self.placeholders['labels_mask'])
        # VAT_loss
        self.loss += 0.1 * masked_softmax_KL(self.outputs, self.outputs2,
                                                       1 - self.placeholders['labels_mask'])

        # add 隐藏层loss
        # self.loss += 2 * masked_softmax_KL(self.h1, self.h2,
        #                               self.placeholders['labels_mask'])

    def _accuracy(self):

        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            Trades=self.trades,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            Trades=self.trades,
                                            logging=self.logging))
        self.layers.append(Dense(input_dim=FLAGS.hidden2,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 FPAT=self.FPAT,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

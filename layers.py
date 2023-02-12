from inits import *
import tensorflow as tf2

tf = tf2.compat.v1
tf.disable_v2_behavior()

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False, FPAT=False, Trades=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.FPAT = FPAT

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            if self.FPAT:
                self.vars['weights'] = glorot([input_dim, output_dim],
                                            name='weights')
                self.vars['weights'] = tf.stop_gradient(self.vars['weights'])
            else:
                self.vars['weights'] = glorot([input_dim, output_dim],
                                              name='weights')

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        if x.shape[0] == 2:
            # convolve A'
            # transform
            output = dot(x[0], self.vars['weights'], sparse=self.sparse_inputs)

            # bias
            if self.bias:
                output += self.vars['bias']
            # convolve A
            # transform
            output2 = dot(x[1], self.vars['weights'], sparse=self.sparse_inputs)

            # bias
            if self.bias:
                output2 += self.vars['bias']

            if self.FPAT:
                return self.act(output), x[0], x[1]
            else:
                return self.act(output),  self.act(output2)
        else:
            # transform
            output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

            # bias
            if self.bias:
                output += self.vars['bias']
            return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., FPAT=False, Trades=False,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.ori_support = placeholders['ori_support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.Trades = Trades

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):  # 定义了一个图，所有的变量在图中，如layer[0]_vars
            for i in range(len(self.support)):  # len(support)=1
                # 定义可优化参数矩阵（变量）W，1433 × 16  Glorot 正态分布初始化器
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
                if self.Trades:
                    self.vars['weights_' + str(i)] = tf.stop_gradient(self.vars['weights_' + str(i)])
            #  DAXW没有偏执
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        #  如果第二层Layer #此处保留一个第三项，即A和A'两个隐藏层的特征
        if x.shape[0] == 2:
            # convolve A'
            supports = list()
            for i in range(len(self.support)):
                if not self.featureless:
                    pre_sup = dot(x[0], self.vars['weights_' + str(i)],  # weights_0即W，求得XW
                                  sparse=self.sparse_inputs)
                else:
                    pre_sup = self.vars['weights_' + str(i)]
                support = dot(self.support[i], pre_sup, sparse=self.sparse_inputs)  # h=LXW
                supports.append(support)  # support(h)加入output
            output = tf.add_n(supports)  # 把supports按照每一个support格式相加
            #h1 = x[0]
            # convolve A
            ori_supports = list()
            for i in range(len(self.ori_support)):
                if not self.featureless:
                    pre_sup = dot(x[1], self.vars['weights_' + str(i)],  # weights_0即W，求得XW
                                  sparse=self.sparse_inputs)
                else:
                    pre_sup = self.vars['weights_' + str(i)]
                ori_support = dot(self.ori_support[i], pre_sup, sparse=self.sparse_inputs)  # h=LXW
                ori_supports.append(ori_support)  # support(h)加入output
            output2 = tf.add_n(ori_supports)  # 把supports按照每一个support格式相加
            #h2 = x[1]
        else:  # 第一层Layer
            # convolve A'
            supports = list()
            for i in range(len(self.support)):
                if not self.featureless:
                    pre_sup = dot(x, self.vars['weights_' + str(i)],  # weights_0即W，求得XW
                                  sparse=self.sparse_inputs)
                else:
                    pre_sup = self.vars['weights_' + str(i)]
                support = dot(self.support[i], pre_sup, sparse=self.sparse_inputs)  # h=LXW
                supports.append(support)  # support(h)加入output
            output = tf.add_n(supports)  # 把supports按照每一个support格式相加

            # convolve A
            ori_supports = list()
            for i in range(len(self.ori_support)):
                if not self.featureless:
                    pre_sup = dot(x, self.vars['weights_' + str(i)],  # weights_0即W，求得XW
                                  sparse=self.sparse_inputs)
                else:
                    pre_sup = self.vars['weights_' + str(i)]

                ori_support = dot(self.ori_support[i], pre_sup, sparse=self.sparse_inputs)  # h=LXW

                ori_supports.append(ori_support)  # support(h)加入output
            output2 = tf.add_n(ori_supports)  # 把supports按照每一个support格式相加
        # bias
        if self.bias:
            output += self.vars['bias']
            output2 += self.vars['bias']

        if x.shape[0] == 2:
            return self.act(output), self.act(output2)  # , h1, h2
        else:
            return self.act(output),  self.act(output2)
        # return output,output2

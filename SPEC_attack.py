from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()
from utils import construct_feed_dict, bisection, filter_potential_singletons
from models import GCN


class SPECAttack:
    def __init__(self, sess, model, features, epsilon, k, mu, ori_adj, ratio):
        self.sess = sess
        self.model = model
        self.features = features
        self.eps = epsilon
        self.ori_adj = ori_adj
        self.total_edges = np.sum(self.ori_adj) / 2
        self.n_node = self.ori_adj.shape[-1]  # 最后一维的个数，此处指列数，点的个数
        self.mu = mu
        self.xi = 1e-5
        self.ratio = ratio  # 扰动率

    #
    def evaluate(self, support, labels, mask):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(self.features, support, labels, mask, self.model.placeholders)
        feed_dict_val.update({self.model.placeholders['support'][i]: support[i] for i in range(len(support))})
        outs_val = self.sess.run([self.model.attack_loss, self.model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    def perturb(self, feed_dict, discrete, y_test, test_mask, k, eps=None, ori_support=None):
        if self.ratio == 0:
            return ori_support

        if eps: self.eps = eps  # 总扰动

        for epoch in range(k):

            t = time.time()
            feed_dict.update({self.model.placeholders['mu']: self.mu / np.sqrt(epoch + 1)})

            # s \in [0,1]
            a, support, modified_adj = self.sess.run(
                [self.model.a, self.model.placeholders['support'], self.model.modified_A], feed_dict=feed_dict)
            modified_adj = np.array(modified_adj[0])
            upper_S_update = bisection(a, self.eps, self.xi)
            #  S是一个扰动矩阵，连续型的
            feed_dict.update({self.model.placeholders['s'][i]: upper_S_update[i] for i in range(len(upper_S_update))})

            if discrete:  # 最后需要一个离散矩阵
                upper_S_update_tmp = upper_S_update[:]
                if epoch == k - 1:
                    acc_record, support_record, p_ratio_record = [], [], []
                    #  最后一个轮次开始采样，决定扰动边缘
                    print('last round, perturb edges by probabilities!')
                    for i in range(10):
                        randm = np.random.uniform(size=(self.n_node, self.n_node))  # 取一个nxn的random矩阵
                        upper_S_update = np.where(upper_S_update_tmp > randm, 1, 0)  # 采样一次，决定哪个位置进行扰动
                        #  更新扰动矩阵
                        feed_dict.update(
                            {self.model.placeholders['s'][i]: upper_S_update[i] for i in range(len(upper_S_update))})
                        # run一次，获得新的support_d
                        a, support_d, modified_adj_d = self.sess.run(
                            [self.model.a, self.model.placeholders['support'], self.model.modified_A],
                            feed_dict=feed_dict)
                        # modified_adj_d = np.array(modified_adj_d[0])
                        # plt.plot(np.sort(upper_S_update[np.nonzero(upper_S_update)]))
                        cost, acc, duration = self.evaluate(support_d, y_test, test_mask)
                        pr = np.count_nonzero(upper_S_update[0]) / self.total_edges
                        if pr <= self.ratio:  # 在规定范围内进行记录
                            acc_record.append(acc)
                            support_record.append(support_d)
                            p_ratio_record.append(pr)
                    print("Step:", '%04d' % (epoch + 1), "test_loss=", "{:.5f},".format(cost),
                          "test_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
                    if len(acc_record) > 0:
                        support_d = support_record[np.argmin(np.array(acc_record))]
                    break
            cost, acc, duration = self.evaluate(support, y_test, test_mask)

            # Print results
            if epoch == k - 1 or epoch == 0:
                print("Step:", '%04d' % (epoch + 1), "test_loss=", "{:.5f}".format(cost),
                      "test_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

            # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            #     print("Early stopping...")
            #     break
        # if discrete:
        #     print("perturb ratio", np.count_nonzero(upper_S_update[0])/self.total_edges)
        # else:
        #     print("perturb ratio (count by L1)", np.sum(upper_S_update[0])/self.total_edges)

        # return modified_adj_d,feed_dict if discrete else modified_adj,feed_dict
        return support_d if discrete else support

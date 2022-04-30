#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of GLISTA --- LISTA with learned gate.
    @inproceedings{wu2019sparse,
  title={Sparse Coding with Gated Learned ISTA},
  author={Wu, Kailun and Guo, Yiwen and Li, Ziang and Zhang, Changshui},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
"""

import tensorflow as tf
from utils.tf import shrink_ss
import numpy as np
from utils.tf import shrink_ss_return_index
from utils.tf import shrink_free_return_index

class LM_GLISTA_en_de(tf.Module):
    def __init__(self, _layers, _A, _unshared, _nums, _ss, _percent, _max_percent, _uf,
                 _alti, _overshoot, _gain, _gain_fun, _over_fun, _both_gate, _T_combine, _T_middle,
                 _en_de, _D_cl2num, _classify_type, _classes, _Vscope,
                 _lam=1., soft=None):
        super(LM_GLISTA_en_de, self).__init__()
        self.lays = _layers
        self.A = _A
        self.M = self.A.shape[0]
        self.N = self.A.shape[1]
        self.is_unshared = _unshared
        self.nums = _nums
        self.has_ss = _ss
        self.p = _percent
        self.maxp = _max_percent
        self.uf = _uf  # str
        self.alti = _alti
        self.overshoot = _overshoot
        self.gain = _gain  # bool
        self.gain_fun = _gain_fun  # str
        self.over_fun = _over_fun
        self.both_gate = _both_gate  # bool
        self.T_combine = _T_combine
        self.T_middle = _T_middle  # int  10
        self.logep = -2.0
        self._ps = [(t + 1) * self.p for t in range(self.lays)]
        self._ps = np.clip(self._ps, 0.0, self.maxp)
        self.is_en_de = _en_de
        self.D_cl2num = _D_cl2num
        self.classify_type = _classify_type
        self.classes = _classes
        self.Vscope = _Vscope
        self.lam = _lam

        self.A_ = tf.constant(self.A, dtype=tf.float32)
        self.L_ = tf.pow(tf.reduce_max(tf.linalg.svd(self.A)[0]), 2)
        self.W_ = tf.transpose(self.A) / self.L_
        self.theta_ = self.lam / self.L_
        self.S_ = tf.eye(self.N) - tf.matmul(tf.transpose(self.A), self.A) / self.L_
        self.theta = np.ones((self.N, 1), dtype=np.float32) * self.theta_

        self.gain_gate = []
        self.over_gate = []
        if self.both_gate:
            for i in range(0, self.lays):
                self.over_gate.append(self.over_fun)
                if self.gain_fun == 'combine':
                    if i > self.T_combine:
                        self.gain_gate.append('inv')
                        # self.over_gate.append('none')
                    else:
                        self.gain_gate.append('relu')  ##2
                        # self.over_gate.append(self.over_fun)
                else:
                    self.gain_gate.append(self.gain_fun)
        else:
            for i in range(0, self.lays):
                if i > self.T_middle:
                    self.gain_gate.append(self.gain_fun)
                    self.over_gate.append('none')
                else:
                    self.gain_gate.append('none')
                    self.over_gate.append(self.over_fun)
        print(self.gain_gate)
        print(self.over_gate)

        if self.uf == 'combine' and self.gain:
            self.combine_function = True
        else:
            self.combine_function = False

        if soft is None:
            self.soft = lambda p, th: tf.multiply(tf.sign(p), tf.maximum(tf.abs(p) - th, 0))
        else:
            self.soft = soft
        self.set_layers()

    def set_layers(self):
        self.Ss_ = []
        self.Ws_ = []
        self.thetas_ = []
        self.D_ = []
        self.D_over = []
        self.W_g_ = []
        self.B_g_ = []
        self.b_g_ = []
        self.log_epsilon_ = []
        self.alti_ = []
        self.alti_over = []
        self.vars_in_layer = []

        B_g = (tf.transpose(self.A_) / self.L_)  #
        W_g = tf.eye(self.N) - tf.matmul(self.W_, self.A_)
        b_g = tf.zeros((self.N, 1), dtype=tf.float32)
        D = tf.ones((self.N, 1), dtype=tf.float32)

        if self.lays >= 2:
            with tf.name_scope(self.Vscope + str(1)):
                self.gamma = tf.Variable(0.0001, name="gamma_%d" % (1), dtype=tf.float32,
                                         trainable=True)

        for i in range(self.lays):
            with tf.name_scope(self.Vscope + str(i)):
                if i % self.nums == 0:
                    '''
                    self.B_g_.append(tf.Variable(tf.initializers.GlorotUniform()(B_g.shape), name='B_g', dtype=tf.float32,
                                            trainable=True))
                    self.b_g_.append(tf.Variable(tf.initializers.GlorotUniform()(b_g.shape), name='b_g', dtype=tf.float32,
                                            trainable=True))
                    self.W_g_.append(tf.Variable(tf.initializers.GlorotUniform()(W_g.shape), name='W_g', dtype=tf.float32,
                                            trainable=True))
                    '''
                    self.Ws_.append(tf.Variable(self.W_, name="W_%d" % (i), dtype=tf.float32, trainable=True))
                    self.Ss_.append(tf.Variable(self.S_, name="S_%d" % (i), dtype=tf.float32, trainable=True))
                    self.alti_.append(tf.Variable(self.alti, name="alti_%d" % (i), dtype=tf.float32, trainable=True))
                    self.D_.append(tf.Variable(D, name='D_%d' % (i), dtype=tf.float32, trainable=True))
                else:
                    self.Ws_.append(self.Ws_[-1])
                    self.Ss_.append(self.Ss_[-1])
                    self.D_.append(self.Ws_[-1])
                    self.alti_.append(self.gammas_[-1])
                if i < 7:
                    _logep = self.logep
                elif i < 10:
                    _logep = self.logep - 2.0
                else:
                    _logeq = -7.0
                self.log_epsilon_.append(tf.Variable(_logep,name='log_epsilon_%d' % (i), dtype=tf.float32, trainable=True))
                self.thetas_.append(tf.Variable(self.theta, name="theta_%d" % (i), dtype=tf.float32, trainable=True))

        self.vars_in_layer = list (zip (self.log_epsilon_,self.Ws_,self.Ss_,self.thetas_,self.D_,self.alti_))

    def __call__(self, in_b):
        def reweight_function(x, D, theta, alti_):
            reweight = 1.0 + alti_ * theta * tf.nn.relu(1 - tf.nn.relu(D * tf.abs(x)))
            return reweight

        def reweight_inverse(x, D, theta, alti):
            reweight = 1.0 + alti * theta * 1.0 / (0.1 + tf.abs(D * x))
            return reweight

        def reweight_exp(x, D, theta, alti):
            reweight = 1.0 + alti * theta * tf.exp(-D * tf.abs(x))
            return reweight

        def reweight_sigmoid(x, D, theta, alti):
            reweight = 1.0 + alti * theta * tf.nn.sigmoid(-D * tf.abs(x))
            return reweight

        def reweight_inverse_variant(x, D, theta, alti, epsilon):
            reweight = 1.0 + alti * theta * 1.0 / (epsilon + tf.abs(D * x))
            return reweight

        def gain(x, D, theta, alti_, epsilon, gain_fun):
            if gain_fun == 'relu':
                use_function = reweight_function
            if gain_fun == 'inv':
                use_function = reweight_inverse
            elif gain_fun == 'exp':
                use_function = reweight_exp
            elif gain_fun == 'sigm':
                use_function = reweight_sigmoid
            elif gain_fun == 'inv_v':
                use_function = reweight_inverse_variant
                return use_function(x, D, theta, alti_, epsilon)
            elif gain_fun == 'none':
                return 1.0 + 0.0 * reweight_function(x, D, theta, alti_) + 0.0 * epsilon
            return use_function(x, D, theta, alti_) + 0.0 * epsilon

        def overshoot(alti, Part_1, Part_2):
            if self.overshoot:
                return 1.0 - alti * Part_1 * Part_2
            else:
                return 1.0 + 0.0 * alti * Part_1 * Part_2

        x = tf.zeros(shape=(self.N, in_b.shape[1]), dtype=tf.float32)
        for i in range(self.lays):
            log_epsilon, W, S, theta_, D, alti = self.vars_in_layer[i]
            percent = self._ps[i]
            if i == 0:
                in_ = gain(x, D, theta_ * 0.0 + 1.0, alti, tf.exp(log_epsilon), self.gain_gate[i])
                Part_2_inv = theta_
            else:
                in_ = gain(x, D, theta_p * 0.0 + 1.0, alti, tf.exp(log_epsilon), self.gain_gate[i])
                Part_2_inv = theta_p
                # This is the bound of layer of ReLU and inverse function.
            # res_ = in_b - tf.matmul(self._kA_, in_ * x)
            if self.has_ss:
                percent = self._ps[i]
                xh_title, cindex = shrink_ss_return_index(in_ * tf.matmul(S, x) + tf.matmul(W, in_b), theta_, percent)
            else:
                xh_title, cindex = shrink_free_return_index(in_ * tf.matmul(S, x) + tf.matmul(W, in_b), theta_, percent)
            theta_p = theta_ * cindex
            x = xh_title
        if not self.is_en_de:
            return x
        else:
            if self.classify_type == "SRC":
                # classifying
                class_resi = []  ###(classes, batch_size)
                for name, num in self.D_cl2num.items():
                    # get the block D(i)
                    Di = tf.gather(self.A, num, axis=1)  # Di.shape = (120,len(A_dict[name]))
                    xi = tf.gather(x, num, axis=0)  # xi.shape = (len(A_dict[name]),batch_size)
                    yi = tf.matmul(Di, xi)  # yi.shape = (120, batch_size)
                    # compute the residuals r = ||in_y - Di*xi||2
                    resi = tf.linalg.norm((in_b - yi), axis=0)  # residuals.shape(batch_size,)
                    class_resi.append(resi)
                class_resi = tf.transpose(tf.cast(
                    1.0 - tf.constant(np.array(class_resi), dtype=tf.float32) / tf.reduce_sum(class_resi, axis=0),
                    dtype=tf.float32))
                # class_resi = tf.nn.softmax(-tf.constant(np.array(class_resi), dtype=tf.float32), axis=0)  # (classes, batch_size)

                return x, class_resi
            # elif self.classify_type == "dense":
            #     class_classify = tf.nn.softmax(tf.matmul(self.Wd, x), axis=0)  # (classes, batch_size)
            #
            #     return x, class_classify
            #
            # elif self.classify_type == "Cosine_simi":
            #     class_cos = []
            #     y_n = tf.linalg.norm(in_b, axis=0)  # ||iny_n||
            #     for name, num in self.D_cl2num.items():
            #         # get the block D(i)
            #         Di = tf.gather(self.D, num, axis=1)  # Di.shape = (120,len(A_dict[name]))
            #         xi = tf.gather(x, num, axis=0)  # xi.shape = (len(A_dict[name]),batch_size)
            #         yi = tf.matmul(Di, xi)  # yi.shape = (120, batch_size)
            #
            #         yi_si = tf.reduce_sum(tf.math.multiply(in_b, yi), axis=0)  # in_y.T.dot(yi)
            #         yi_n = tf.linalg.norm(yi, axis=0)  # ||yi_n||
            #         si_n = y_n * yi_n  # ||iny_n||*||yi_n||
            #         si = yi_si / si_n  # in_y.yi/||iny_n||*||yi_n||
            #         class_cos.append(si)
            #
            #     class_cos = tf.transpose(tf.cast(np.array(class_cos), dtype=tf.float32))  # (batch_size, classes)
            #     return x, class_cos
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of TiLISTA --- LISTA with tied weight.
    @inproceedings{
    liu2018alista,
    title={{ALISTA}: Analytic Weights Are As Good As Learned Weights in {LISTA}},
    author={Jialin Liu and Xiaohan Chen and Zhangyang Wang and Wotao Yin},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=B1lnzn0ctQ},
    }
"""

import tensorflow as tf
from utils.tf import shrink_ss
import numpy as np
from utils.tf import shrink_ss
from utils.tf import shrink_free

class TiLISTA_en_de(tf.Module):
    def __init__(self, _layers, _A, _unshared, _nums, _ss, _percent, _max_percent, _en_de, _D_cl2num, _classify_type, _classes, _Vscope,
                 _lam=1., soft=None):
        super(TiLISTA_en_de, self).__init__()
        """
            _layers: int
            _A: matrix
            _unshared: bool, whether weights are shared through layers or not
            _nums: int, for controlling shared weights
            _ss: bool, whether support support selection
            _percent: float, Percent of entries to be selected as support in each layer
            _max_percent: float, Maximum percentage of entries to be selected as support in each layer
            _na: bool, if add Nesterov Acceleration
            _en_de: bool, true for face recognition
            _D_cl2num: todo 
            _classify_type: str
            _classes: int, how many classes
            _Vscope: str, variants name scope
            _lam: float

            A_: matrix A
            L: step size, the reciprocal of the largest singular square value. 
            W_: A^T/L
            theta_: lambda/L

        """

        self.lays = _layers
        self.A = _A
        self.M = self.A.shape[0]
        self.N = self.A.shape[1]
        self.is_unshared = _unshared
        self.nums = _nums
        self.has_ss = _ss
        self.p = _percent
        self.maxp = _max_percent

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
        self.set_layers()

    def set_layers(self):
        self.Ws_ = []
        self.alphas_ = []
        self.thetas_ = []
        self.vars_in_layer = []

        # network design, does not need shared weight for every layer
        for i in range(self.lays):
            with tf.name_scope(self.Vscope + str(i)):
                if i % self.nums == 0:
                    self.Ws_.append(tf.Variable(self.W_, name="W_%d" % (i), dtype=tf.float32, trainable=True))
                else:
                    self.Ws_.append(self.Ws_[-1])

                self.alphas_.append(tf.Variable(1.0, name="alpha_%d" % (i), dtype=tf.float32, trainable=True))
                self.thetas_.append(tf.Variable(self.theta_, name="theta_%d" % (i), dtype=tf.float32, trainable=True))

        self.vars_in_layer = list(zip(self.Ws_, self.alphas_, self.thetas_))

    def __call__(self, in_b):
        x = tf.zeros(shape=(self.N, in_b.shape[1]), dtype=tf.float32)
        for i in range(self.lays):
            W, alpha, T = self.vars_in_layer[i]
            res = in_b - tf.matmul(self.A_, x)
            if self.has_ss:
                percent = self._ps[i]
                x = shrink_ss(x + alpha * tf.matmul(W, res), T, percent)
            else:
                x = shrink_free(x + alpha * tf.matmul(W, res), T)
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
            #         Di = tf.gather(self.A, num, axis=1)  # Di.shape = (120,len(A_dict[name]))
            #         xi = tf.gather(x, num, axis=0)  # xi.shape = (len(A_dict[name]),batch_size)
            #         yi = tf.matmul(Di, xi)  # yi.shape = (120, batch_size)
            #
            #         yi_si = tf.reduce_sum(tf.math.multiply(in_b, yi), axis=0)  # in_y.T.dot(yi)
            #         yi_n = tf.linalg.norm(yi, axis=0)  # ||yi_n||
            #         si_n = y_n * yi_n  # ||iny_n||*||yi_n||
            #         si = yi_si / (si_n + 0.0001)  # in_y.yi/||iny_n||*||yi_n||
            #         class_cos.append(si)
            #
            #     class_cos = tf.transpose(tf.cast(np.array(class_cos), dtype=tf.float32))  # (batch_size, classes)
            #     return x, class_cos
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of ALISTA --- LISTA with analytic weight.
    @inproceedings{liu2019alista,
  title={ALISTA: Analytic weights are as good as learned weights in LISTA},
  author={Liu, Jialin and Chen, Xiaohan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019}
}
"""

import tensorflow as tf
from utils.tf import shrink_ss
import numpy as np
from utils.tf import shrink_ss
from utils.tf import shrink_free

class ALISTA_en_de(tf.Module):
    def __init__(self, _layers, _A, _W, _unshared, _nums, _ss, _percent, _max_percent, _na, _en_de, _D_cl2num, _classify_type, _classes, _Vscope,
                 _lam=1., soft=None):
        super(ALISTA_en_de, self).__init__()
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
        self.W = _W
        self.M = self.A.shape[0]
        self.N = self.A.shape[1]
        self.is_unshared = _unshared
        self.nums = _nums
        self.has_ss = _ss
        self.p = _percent
        self.maxp = _max_percent

        self._ps = [(t + 1) * self.p for t in range(self.lays)]
        self._ps = np.clip(self._ps, 0.0, self.maxp)

        self.is_na = _na
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
        self.alphas_ = []
        self.thetas_ = []
        self.vars_in_layer = []

        # pre-calculated W
        if not tf.is_tensor(self.W):
            self._W_ = tf.constant(value=self.W, dtype=tf.float32)
        else:
            self._W_ = self.W

        self._Wt_ = tf.transpose(self._W_, perm=[1, 0])

        for i in range(self.lays):
            with tf.name_scope(self.Vscope + str(i)):
                self.alphas_.append(tf.Variable(1.0, name="alpha_%d" % (i), dtype=tf.float32, trainable=True))
                self.thetas_.append(tf.Variable(self.theta_, name="theta_%d" % (i), dtype=tf.float32, trainable=True))

        self.vars_in_layer = list(zip(self.alphas_, self.thetas_))

    def __call__(self, in_b):
        x = tf.zeros(shape=(self.N, in_b.shape[1]), dtype=tf.float32)
        xs = x
        # add Nesterov Acceleration
        if self.is_na:
            for i in range(self.lays):
                alpha, T = self.vars_in_layer[i]
                if i == 0:
                    t = tf.cast(1, dtype=tf.float32)
                    z = xs
                else:
                    ts = t
                    t = (1. / 2.) * (1. + tf.math.sqrt(1. + 4. * ts * ts))
                    z = x + (ts - 1) / t * (x - xs)
                    # keep x(k-1)
                    xs = x
                res = in_b - tf.matmul(self.A_, z)
                if self.has_ss:
                    percent = self._ps[i]
                    x = shrink_ss(x + alpha * tf.matmul(self._Wt_, res), T, percent)
                else:
                    x = shrink_free(x + alpha * tf.matmul(self._Wt_, res), T)
        else:
            for i in range(self.lays):
                alpha, T = self.vars_in_layer[i]
                res = in_b - tf.matmul(self.A_, x)
                if self.has_ss:
                    percent = self._ps[i]
                    x = shrink_ss(x + alpha * tf.matmul(self._Wt_, res), T, percent)
                else:
                    x = shrink_free(x + alpha * tf.matmul(self._Wt_, res), T)
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
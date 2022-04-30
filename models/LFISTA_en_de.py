#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
Implementation of LFISTA
    @article{parisunderstanding,
      title={Understanding Trainable Sparse Coding with Matrix Factorization},
      author={Thomas Moreau CMLA-ENS and Bruna, J and Audiffren, J}
}

'''

import tensorflow as tf
import numpy as np

class LFISTA_en_de(tf.Module):
    def __init__(self, _layers, _A, _unshared, _nums, _en_de, _D_cl2num, _classify_type, _classes, _Vscope,
                 _lam=1., soft=None):
        super(LFISTA_en_de, self).__init__()
        """
            _layers: int
            _A: matrix
            _unshared: bool, whether weights are shared through layers or not
            _nums: int, for controlling shared weights
            _en_de: bool, true for face recognition
            _D_cl2num: todo 
            _classify_type: str
            _classes: int, how many classes
            _Vscope: str, variants name scope
            _lam: float

            L: step size, the reciprocal of the largest singular square value. 
            W_: A^T/L
            theta_: lambda/L
            S_: I - A^T*A/L

        """
        self.lays = _layers
        self.A = _A
        self.M = self.A.shape[0]
        self.N = self.A.shape[1]
        self.is_unshared = _unshared
        self.nums = _nums
        self.is_en_de = _en_de
        self.D_cl2num = _D_cl2num
        self.classify_type = _classify_type
        self.classes = _classes
        self.Vscope = _Vscope
        self.lam = _lam

        self.L_ = tf.pow(tf.reduce_max(tf.linalg.svd(self.A)[0]), 2)
        self.W_ = tf.transpose(self.A) / self.L_
        self.theta_ = self.lam / self.L_
        self.S_ = tf.eye(self.N) - tf.matmul(tf.transpose(self.A), self.A) / self.L_

        # generate t(k)
        self.Ks_ = []
        self.ks = tf.constant(1, dtype=tf.float32)
        self.Ks_.append(self.ks)
        for _ in range(self.lays - 1):
            self.k = (1. / 2.) * (1. + tf.math.sqrt(1. + 4. * self.ks * self.ks))
            self.Ks_.append(self.k)
            self.ks = self.k

        if soft is None:
            self.soft = lambda p, th: tf.multiply(tf.sign(p), tf.maximum(tf.abs(p) - th, 0))
        else:
            self.soft = soft
        self.set_layers()

    def set_layers(self):
        self.Wg_ = []
        self.Wm_ = []
        self.Ws_ = []
        self.Ts_ = []
        self.vars_in_layer = []

        with tf.name_scope(self.Vscope + "0"):
            self.W = tf.Variable(self.W_, name="W_%d" % (0), dtype=tf.float32, trainable=True)
            self.theta = tf.Variable(self.theta_, name="theta_%d" % (0), dtype=tf.float32, trainable=True)

        for i in range(self.lays-1):
            with tf.name_scope(self.Vscope + str(i+1)):
                ks = self.Ks_[i]
                k = self.Ks_[i + 1]
                if i % self.nums == 0:
                    self.Ws_.append(tf.Variable(self.W_, name="W_%d" % (i+1), dtype=tf.float32, trainable=True))
                else:
                    self.Ws_.append(self.Ws_[-1])

                self.Wg_.append(tf.Variable((1 + (ks - 1) / k) * self.S_, name="Wg_%d" % (i + 1), dtype=tf.float32, trainable=True))
                self.Wm_.append(tf.Variable(((1 - ks) / k) * self.S_, name="Wm_%d" % (i + 1), dtype=tf.float32, trainable=True))
                self.Ts_.append(tf.Variable(self.theta_, name="theta_%d" % (i+1), dtype=tf.float32, trainable=True))

        self.vars_in_layer = list(zip(self.Wg_, self.Wm_, self.Ws_, self.Ts_))

    def __call__(self, in_b):
        x = self.soft(tf.matmul(self.W, in_b), self.theta)
        xxs = tf.zeros(shape=x.shape, dtype=tf.float32)
        for i in range(self.lays-1):
            Wg, Wm, We, Ts = self.vars_in_layer[i]
            xs = x
            x = self.soft(tf.matmul(Wg, xs) + tf.matmul(Wm, xxs) + tf.matmul(We, in_b), Ts)
            xxs = xs
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
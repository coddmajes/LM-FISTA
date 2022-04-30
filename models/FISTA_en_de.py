#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Implementation of FISTA
    @article{beck2009fast,
        title={A fast iterative shrinkage-thresholding algorithm for linear inverse problems},
        author={Beck, Amir and Teboulle, Marc},
        journal={SIAM journal on imaging sciences},
        volume={2},
        number={1},
        pages={183--202},
        year={2009},
        publisher={SIAM}
}

'''

import tensorflow as tf
import numpy as np

class FISTA_en_de(tf.Module):
    def __init__(self, _layers, _A, _unshared, _nums, _en_de, _D_cl2num, _classify_type, _classes, _Vscope,
                 _lam=1., soft=None):
        super(FISTA_en_de, self).__init__()
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

        if soft is None:
            self.soft = lambda p, th: tf.multiply(tf.sign(p), tf.maximum(tf.abs(p) - th, 0))
        else:
            self.soft = soft

        # for face recognition
        if self.classify_type == "dense":
            with tf.name_scope(self.Vscope + "0"):
                self.Wd = tf.Variable(tf.initializers.GlorotUniform()((self.classes, self.N)), name="Wd_%d" % (0),
                                      dtype=tf.float32, trainable=True)
        elif self.classify_type == "double_LISTA_net":
            with tf.name_scope(self.Vscope + "99"):
                self.Wd = tf.Variable(tf.initializers.GlorotUniform()((self.classes, self.N)), name="Wd_%d" % (0),
                                      dtype=tf.float32, trainable=True)

    def __call__(self, in_b):
        xs = tf.zeros (shape=(self.N, in_b.shape[1]), dtype=tf.float32)
        for i in range(self.lays):
            if i == 0:
                t = 1.0
                z = xs
            else:
                ts = t
                t = (1. / 2.) * (1. + tf.math.sqrt(1. + 4. * ts * ts))
                z = x + (ts - 1) / t * (x - xs)
                # keep x(k-1)
                xs = x
            x = self.soft(tf.matmul(self.W_, in_b) + tf.matmul(self.S_, z), self.theta_)
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
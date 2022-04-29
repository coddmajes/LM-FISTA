#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : utils/tf.py
author: Xiaohan Chen
email : chernxh@tamu.edu
date  : 2019-02-18

Utility functions implemented in TensorFlow, including:
    - miscellaneous
    - shrinkage functions
    - circular padding
    - activations
    - subgradient functions
    - related functions
"""

import tensorflow as tf
import tensorflow_probability as tfp


############################################################
#######################   Misc   ###########################
############################################################
def is_tensor(x):
    return isinstance(x, (tf.Tensor, tf.SparseTensor, tf.Variable))


############################################################
####################   Shrinkage   #########################
############################################################
def shrink(input_, theta_):
    """
    Soft thresholding function with input input_ and threshold theta_.
    """
    theta_ = tf.maximum( theta_, 0.0 )
    return tf.sign(input_) * tf.maximum( tf.abs(input_) - theta_, 0.0 )


def shrink_free(input_, theta_):
    """
    Soft Shrinkage function without the constraint that the thresholds must be
    greater than zero.
    """
    return tf.sign(input_) * tf.maximum( tf.abs(input_) - theta_, 0.0 )


def shrink_ss(inputs_, theta_, q):
    """
    Special shrink that does not apply soft shrinkage to entries of top q%
    magnitudes.

    :inputs_: TODO
    :thres_: TODO
    :q: TODO
    :returns: TODO

    """
    abs_ = tf.abs(inputs_)
    thres_ = tfp.stats.percentile(
            abs_, 100.0-q, axis=0, keep_dims=True)

    """
    Entries that are greater than thresholds and in the top q% simultnaneously
    will be selected into the support, and thus will not be sent to the
    shrinkage function.
    """
    index_ = tf.logical_and(abs_ > theta_, abs_ > thres_)
    index_ = tf.cast(index_, tf.float32)
    """Stop gradient at index_, considering it as constant."""
    index_ = tf.stop_gradient(index_)
    cindex_ = 1.0 - index_ # complementary index

    return (tf.multiply(index_, inputs_) +
            shrink_free(tf.multiply(cindex_, inputs_), theta_ ))

def shrink_free_return_index(inputs_, theta_, q):
    """
    Special shrink that does not apply soft shrinkage to entries of top q%
    magnitudes.

    :inputs_: TODO
    :thres_: TODO
    :q: TODO
    :returns: TODO

    """
    abs_ = tf.abs(inputs_)
    thres_ = tfp.stats.percentile(
            abs_, 100.0-q, axis=0, keep_dims=True)

    """
    Entries that are greater than thresholds and in the top q% simultnaneously
    will be selected into the support, and thus will not be sent to the
    shrinkage function.
    """
    index_ = tf.logical_and(abs_ > theta_, abs_ > thres_)
    index_ = tf.cast(index_, tf.float32)
    """Stop gradient at index_, considering it as constant."""
    index_ = tf.stop_gradient(index_)
    cindex_ = 1.0 - index_ # complementary index

    return tf.sign(inputs_) * tf.maximum( tf.abs(inputs_) - theta_, 0.0 ), index_

def shrink_ss_return_index(inputs_, theta_, q):
    """
    Special shrink that does not apply soft shrinkage to entries of top q%
    magnitudes.

    :inputs_: TODO
    :thres_: TODO
    :q: TODO
    :returns: TODO

    """
    abs_ = tf.abs(inputs_)
    thres_ = tfp.stats.percentile(
            abs_, 100.0-q, axis=0, keep_dims=True)

    """
    Entries that are greater than thresholds and in the top q% simultnaneously
    will be selected into the support, and thus will not be sent to the
    shrinkage function.
    """
    index_ = tf.logical_and(abs_ > theta_, abs_ > thres_)
    index_ = tf.cast(index_, tf.float32)
    """Stop gradient at index_, considering it as constant."""
    index_ = tf.stop_gradient(index_)
    cindex_ = 1.0 - index_ # complementary index

    return (tf.multiply(index_, inputs_) +
            shrink_free(tf.multiply(cindex_, inputs_), theta_ )), index_

def shrink_lamp(r_, rvar_, lam_):
    """
    Implementation of thresholding neuron in Learned AMP model.
    """
    theta_ = tf.maximum(tf.sqrt(rvar_) * lam_, 0.0)
    xh_    = tf.sign(r_) * tf.maximum(tf.abs(r_) - theta_, 0.0)
    return xh_
    # shrink = tf.abs(r_) - theta_
    # xh_    = tf.sign(r_) * tf.maximum(shrink, 0.0)
    # xhl0_  = tf.reduce_mean(tf.to_float(shrink>0), axis=0)
    # return xh_, xhl0_
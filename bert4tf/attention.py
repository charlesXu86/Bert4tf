# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   attention.py
 
@Time    :   2019-12-11 23:06
 
@Desc    :
 
'''

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

from bert4tf.layer import Layer


class AttentionLayer(Layer):

    class Params(Layer.Params):
        num_heads         = None
        size_per_head     = None
        initializer_range = 0.02
        query_activation  = None
        key_activation    = None
        value_activation  = None
        attention_dropout = 0.1
        negative_infinity = -10000.0  # used for attention scores before softmax

# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   embeddings.py
 
@Time    :   2019-12-11 22:21
 
@Desc    :
 
'''

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import params_flow as pf

from tensorflow import keras
from tensorflow.keras import backend as K

import bert4tf

class PositionEmbeddingLayer(bert4tf.Layer):
    class Params(bert4tf.Layer.Params):
        max_position_embeddings = 512
        hidden_size = 128


class EmbeddingsProjector(bert4tf.Layer):
    class Params(bert4tf.Layer.Params):
        hidden_size                  = 768
        embedding_size               = None   # None for BERT, not None for ALBERT
        project_embeddings_with_bias = True   # in ALBERT - True for Google, False for brightmart/albert_zh

class BertEmbeddingsLayer(bert4tf.Layer):
    class Params(PositionEmbeddingLayer.Params,
                 EmbeddingsProjector.Params):
        vocab_size               = None
        use_token_type           = True
        use_position_embeddings  = True
        token_type_vocab_size    = 2
        hidden_size              = 768
        hidden_dropout           = 0.1

        extra_tokens_vocab_size  = None  # size of the extra (task specific) token vocabulary (using negative token ids)

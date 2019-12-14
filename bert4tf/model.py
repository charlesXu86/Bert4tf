# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   model.py
 
@Time    :   2019-12-11 22:20
 
@Desc    :
 
'''

from __future__ import absolute_import, division, print_function

from tensorflow import keras
import params_flow as pf

from bert4tf.layer import Layer
from bert4tf.embeddings import BertEmbeddingsLayer
from bert4tf.transformer import TransformerEncoderLayer


class BertModelLayer(Layer):
    """
        Implementation of BERT (arXiv:1810.04805), adapter-BERT (arXiv:1902.00751) and ALBERT (arXiv:1909.11942).

        See: https://arxiv.org/pdf/1810.04805.pdf - BERT
             https://arxiv.org/pdf/1902.00751.pdf - adapter-BERT
             https://arxiv.org/pdf/1909.11942.pdf - ALBERT

        """

    class Params(BertEmbeddingsLayer.Params,
                 TransformerEncoderLayer.Params):
        pass

    def _construct(self, params: Params):
        self.embeddings_layer = None
        self.encoders_layer = None

        self.supports_masking = True

    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
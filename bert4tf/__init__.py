# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   __init__.py.py
 
@Time    :   2019-11-03 15:07
 
@Desc    :
 
'''

from __future__ import division, absolute_import, print_function

from .version import __version__

from . import modeling
from . import optimization
# from . import extract_features

from .layer import Layer


from .tokenization import bert_tokenization
from .tokenization import albert_tokenization

from .model import BertModelLayer
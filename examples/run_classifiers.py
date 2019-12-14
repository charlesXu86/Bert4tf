# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   run_classifiers.py
 
@Time    :   2019-11-28 20:36
 
@Desc    :
 
'''

import os
import re
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import math

from tqdm import tqdm
from bert4tf import BertModelLayer
from bert4tf.tokenization.bert_tokenization import FullTokenizer

from bert4tf.loader_bert import StockBertConfig, map_stock_config_to_params, load_stock_weights

if tf.__version__.startswith('1.'):
    tf.enable_eager_execution = True



# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in tqdm(os.listdir(directory), desc=os.path.basename(directory)):
        with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True)

    train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                         "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                        "aclImdb", "test"))

    return train_df, test_df

class MovieReviewData:
    DATA_COLUMN = "sentence"
    LABEL_COLUMN = "polarity"

    def __init__(self, tokenizer: FullTokenizer, sample_size=None, max_seq_len=128):
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.max_seq_len = 0
        train, test = download_and_load_datasets()

        train, test = map(lambda df: df.reindex(df[MovieReviewData.DATA_COLUMN].str.len().sort_values().index),
                          [train, test])

        if sample_size is not None:
            assert sample_size % 128 == 0
            train, test = train.head(sample_size), test.head(sample_size)
            # train, test = map(lambda df: df.sample(sample_size), [train, test])

        ((self.train_x, self.train_y),
         (self.test_x, self.test_y)) = map(self._prepare, [train, test])

        print("max seq_len", self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        ((self.train_x, self.train_x_token_types),
         (self.test_x, self.test_x_token_types)) = map(self._pad,
                                                       [self.train_x, self.test_x])

    def _prepare(self, df):
        x, y = [], []
        with tqdm(total=df.shape[0], unit_scale=True) as pbar:
            for ndx, row in df.iterrows():
                text, label = row[MovieReviewData.DATA_COLUMN], row[MovieReviewData.LABEL_COLUMN]
                tokens = self.tokenizer.tokenize(text)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                self.max_seq_len = max(self.max_seq_len, len(token_ids))
                x.append(token_ids)
                y.append(int(label))
                pbar.update()
        return np.array(x), np.array(y)

    def _pad(self, ids):
        x, t = [], []
        token_type_ids = [0] * self.max_seq_len
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
            t.append(token_type_ids)
        return np.array(x), np.array(t)


bert_model_dir = '/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12'
bert_ckpt_file = os.path.join(bert_model_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_model_dir, "bert_config.json")


tokenizer = FullTokenizer(vocab_file=os.path.join(bert_model_dir, "vocab.txt"))
data = MovieReviewData(tokenizer,
                       sample_size=10*128*2,#5000,
                       max_seq_len=128)

print("            train_x", data.train_x.shape)
print("train_x_token_types", data.train_x_token_types.shape)
print("            train_y", data.train_y.shape)

print("             test_x", data.test_x.shape)

print("        max_seq_len", data.max_seq_len)


def flatten_layers(root_layer):
    if isinstance(root_layer, keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sublayer in flatten_layers(layer):
            yield sublayer

def freeze_bert_layers(l_bert):
    '''
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    :param l_bert:
    :return:
    '''
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        elif len(layer._layers) == 0:
            layer.trainable = False
        l_bert.embeddings_layer.trainable = False

def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):
    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate * math.exp(
                math.log(end_learn_rate / max_learn_rate) * (epoch - warmup_epoch_count + 1) / (
                            total_epoch_count - warmup_epoch_count + 1))
        return float(res)

    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler

def create_model(max_seq_length, adapter_size=64):
    '''

    :param max_seq_length:
    :param adapter_size:
    :return:
    '''

    # create the bert layer
    with tf.io.gfile.GFile(bert_config_file, 'r') as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = adapter_size
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='input_ids')
    # token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")
    # output         = bert([input_ids, token_type_ids])
    output = bert(input_ids)

    print('bert shape', output.shape)
    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation='tanh')(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=2, activation='softmax')(logits)

    # model = keras.Model(inputs=[input_ids, token_type_ids], outputs=logits)
    # model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])
    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_length))

    # 加载预训练模型
    load_stock_weights(bert, bert_ckpt_file)

    # freeze weights if use adapter-bert
    if adapter_size is not None:
        freeze_bert_layers(bert)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')])

    model.summary()

    return model

adapter_size = 0

model = create_model(data.max_seq_len, adapter_size=adapter_size)

pred_sentences = [
    "That movie was absolutely awful",
    "The acting was a bit lacking",
    "The film was creative and surprising",
    "Absolutely fantastic!"
]

pred_tokens = map(tokenizer.tokenize, pred_sentences)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids + [0] * (data.max_seq_len - len(tids)), pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

print('pred_token_ids', pred_token_ids.shape)

res = model.predict(pred_token_ids).argmax(axis=-1)

for text, sentiment in zip(pred_sentences, res):
    print(" text:", text)
    print("  res:", ["negative", "positive"][sentiment])
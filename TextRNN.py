#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2017/11/3 13:34
 @Author  : Kiristingna
 @File    : TextCNN.py
 @Software: PyCharm
"""
# text classify for company
import yaml
import pandas as pd
import jieba
import numpy as np
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input, Flatten, GlobalMaxPooling1D, Bidirectional
from keras.layers.merge import concatenate
from keras.utils import np_utils
from keras.models import model_from_yaml
from keras import backend as K


def process_data():
    df = pd.read_csv('../../data/company/train.csv', index_col=False, header=None)
    df.columns = ['cate', 'content']
    cw = lambda x: list(jieba.cut(x))
    df['words'] = df.content.apply(cw)
    words = df['words'].values
    y = df['cate'].values

    tests = pd.read_csv('../../data/company/test.csv', index_col=False, header=None)
    tests.columns = ['id', 'content']
    cw = lambda x: list(jieba.cut(x))
    tests['words'] = tests.content.apply(cw)
    test = tests['words'].values
    np.save('../../data/company/tokenized.npy', words)
    np.save('../../data/company/label.npy', y)
    np.save('../../data/company/test.npy', test)


def load_data_and_to_vector(window_size=10, n_dims=100, pad_max_length=200):
    '''

    :param window_size: 窗口大小
    :param n_dims: 词向量长度
    :return:
    '''
    words = np.load('../../data/company/tokenized.npy')
    y = np.load('../../data/company/label.npy')
    test = np.load('../../data/company/test.npy')

    total = np.concatenate((words, test))

    w2v = Word2Vec(size=n_dims, window=window_size, workers=4, min_count=1)
    w2v.build_vocab(total)
    w2v.train(total, total_examples=w2v.corpus_count, epochs=w2v.iter)

    _dict = Dictionary()
    # 将一个raw string 转换为根据本词典构构造的向量
    _dict.doc2bow(w2v.wv.vocab.keys(), allow_update=True)
    # w2index is a dict of {word: index} and w2vector is a dict of {word : vector(np.array)}
    w2index = {v: k + 1 for k, v in _dict.items()}  # 词语的索引
    w2vector = {word: w2v[word] for word in w2index.keys()}  # 词语向量

    # 转换序列为索引
    _sequence = []
    for _s in words:
        _sequence.append([w2index[w] for w in _s])

    _tests = []
    for _s in test:
        _tests.append([w2index[w] for w in _s])

    padded_words = sequence.pad_sequences(_sequence, maxlen=pad_max_length)
    padded_test = sequence.pad_sequences(_tests, maxlen=pad_max_length)

    # ------- embed start -------
    n_symbols = len(w2index) + 1  # 因为pad了0
    # every number in index table has n_dims 维的vector
    embedding_weights = np.zeros((n_symbols, n_dims))
    # 填入vector
    for word, index in w2index.items():
        embedding_weights[index, :] = w2vector[word]
    # -------- embed end -------

    return n_symbols, embedding_weights, pad_max_length, padded_words, y, padded_test


def rnn(sequences, y, symbols, weights, max_input_length, vocab_dim=100, n_epochs=5, batch_size=32):
    '''
    :param symbols:
    :param weights:
    :param max_input_length:
    :param vocab_dim:
    :return:
    '''
    y = np_utils.to_categorical(y)
    # input
    input = Input(shape=(max_input_length,))
    # embed layer
    embedding = Embedding(output_dim=vocab_dim, input_dim=symbols,
                        weights=[weights],
                        input_length=max_input_length)(input)

    # 第一双向lSTM层
    bi_lstm_layer = Bidirectional(LSTM(units=100, return_sequences=True), merge_mode='concat')(embedding)

    # 池化层
    pooling_layer = GlobalMaxPooling1D()(bi_lstm_layer)

    # 分类器
    dense = Dense(12, activation='softmax')(pooling_layer)

    trnn = Model(inputs=input, outputs=dense)

    trnn.compile(loss='categorical_crossentropy',
                  optimizer='Adadelta',
                  metrics=['accuracy'])

    trnn.fit(sequences, y, epochs=n_epochs, batch_size=batch_size)

    # 保存
    yaml_string = trnn.to_yaml()
    with open('../../data/company/rnn_model.yaml', 'w') as outfile:
        outfile.write(yaml_string)
        trnn.save_weights('../../data/company/rnn_weights.h5')


def load_and_predict(test_sequence):

    print('load model from disk...')
    with open('../../data/company/rnn_model.yaml', 'r') as f:
        yaml_string = yaml.dump(yaml.load(f))
    model = model_from_yaml(yaml_string)

    print('load weights from disk......')
    model.load_weights('../../data/company/rnn_weights.h5')

    print('rebuild model from disk......')
    model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

    _pros = model.predict(test_sequence)
    _class = _pros.argmax(axis=-1)

    for i in _class:
        print(i)


if __name__ == '__main__':
    # note1: 基于词的模型 要好于基于字的模型
    # note2:

    # 预处理
    # process_data()

    n_symbols, embedding_weights, pad_max_length, sequences, y, tests = load_data_and_to_vector()

    # train model
    rnn(sequences=sequences,
        y=y,
        symbols=n_symbols,
        weights=embedding_weights,
        max_input_length=pad_max_length)

    load_and_predict(tests)

    K.clear_session()

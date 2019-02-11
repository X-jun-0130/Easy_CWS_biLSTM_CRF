# -*- coding:utf-8 -*-
class Parameters(object):

    embedding_size = 100
    vacab_size = 4000
    batch_size = 64
    hidden_dim = 128

    learning_rate = 0.006
    clip = 5.0
    lr = 0.8

    keep_pro = 0.5
    num_tags = 4
    epochs = 8

    train = './data/WordSeg.txt'
    test = './data/test.txt'
    eva = './data/eva.txt'
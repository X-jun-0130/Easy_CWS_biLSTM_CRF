# -*- coding:utf-8 -*-
import re
import pickle
import numpy as np
import tensorflow.contrib.keras as kr
from Parameters import Parameters as pm

state_list = {'B': 0, 'M': 1, 'E': 2, 'S': 3}

#将句子转换为字序列
def get_word(sentence):
    word_list = []
    sentence = ''.join(sentence.split(' '))
    for i in sentence:
        word_list.append(i)
    return word_list

#将句子转换为BMES序列
def get_str(sentence):
    output_str = []
    sentence = re.sub('  ', ' ', sentence) #发现有些句子里面，有两格空格在一起
    list = sentence.split(' ')
    for i in range(len(list)):
        if len(list[i]) == 1:
            output_str.append('S')
        elif len(list[i]) == 2:
            output_str.append('B')
            output_str.append('E')
        else:
            M_num = len(list[i]) - 2
            output_str.append('B')
            output_str.extend('M'* M_num)
            output_str.append('E')
    return output_str


def read_file(filename):
    word, content, label = [], [], []
    text = open(filename, 'r', encoding='utf-8')
    for eachline in text:
        eachline = eachline.strip('\n')
        eachline = eachline.strip(' ')
        word_list = get_word(eachline)
        letter_list = get_str(eachline)
        word.extend(word_list)
        content.append(word_list)
        label.append(letter_list)
    return word, content, label

def word_dict(filename):
    word, _, _ = read_file(filename)
    word = set(word)
    key_dict = {}
    key_dict['<PAD>'] = 0
    key_dict['<UNK>'] = 1
    j = 2
    for w in word:
        key_dict[w] = j
        j += 1
    with open('./data/word2id.pkl', 'wb') as fw:  # 将建立的字典 保存
        pickle.dump(key_dict, fw)
    return key_dict

#key_dict = word_dict('./data/WordSeg.txt')
#print(key_dict)

def sequence2id(filename):
    '''
    :param filename:
    :return: 将文字与标签，转换为数字
    '''
    content2id, label2id = [], []
    _, content, label = read_file(filename)
    with open('./data/word2id.pkl', 'rb') as fr:
        word = pickle.load(fr)
    for i in range(len(label)):
        label2id.append([state_list[x] for x in label[i]])
    for j in range(len(content)):
        w = []
        for key in content[j]:
            if key not in word:
                key = '<UNK>'
            w.append(word[key])
        content2id.append(w)
    return content2id, label2id

def batch_iter(x, y, batch_size = pm.batch_size):
    Len = len(x)
    x = np.array(x)
    y = np.array(y)
    num_batch = int((Len-1) / batch_size) + 1
    indices = np.random.permutation(Len)
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start = i * batch_size
        end = min((i+1) * batch_size, Len)
        yield x_shuffle[start:end], y_shuffle[start:end]

def process(x_batch):
    '''
     :param x_batch: 计算一个batch里面最长句子 长度n
     动态RNN 保持同一个batch里句子长度一致即可，sequence为实际句子长度
     :return: 对所有句子进行padding,长度为n
     '''
    seq_len = []
    max_len = max(map(lambda x: len(x), x_batch))  # 计算一个batch中最长长度
    for i in range(len(x_batch)):
        seq_len.append(len(x_batch[i]))

    x_pad = kr.preprocessing.sequence.pad_sequences(x_batch, max_len, padding='post', truncating='post')
    #y_pad = kr.preprocessing.sequence.pad_sequences(y_batch, max_len, padding='post', truncating='post')

    return x_pad, seq_len



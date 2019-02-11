# -*- coding:utf-8 -*-
import pickle
from data_processing import get_word, process, batch_iter
from Parameters import Parameters as pm
from biLSTM_CRF import biLstm_crf
import tensorflow as tf
import numpy as np

def read_file(filename):
    content = []
    text = open(filename, 'r', encoding='utf-8')
    for eachline in text:
        eachline = eachline.strip('\n')
        eachline = eachline.strip(' ')
        word_list = get_word(eachline)
        content.append(word_list)
    return content

def sequence2id(filename):
    '''
    :param filename:
    :return: 将文字，转换为数字
    '''
    content2id = []
    content = read_file(filename)
    with open('./data/word2id.pkl', 'rb') as fr:
        word = pickle.load(fr)
    for j in range(len(content)):
        w = []
        for key in content[j]:
            if key not in word:
                key = '<UNK>'
            w.append(word[key])
        content2id.append(w)
    return content2id

def convert(sentence, label_line):
    word_cut = ''
    wordlist = get_word(sentence)
    for i in range(len(label_line)):
        if label_line[i] == 2:
            word_cut += wordlist[i]
            word_cut += ' '
        elif label_line[i] == 3:
            word_cut += ' '
            word_cut += wordlist[i]
            word_cut += ' '
        else:
            word_cut += wordlist[i]
    return word_cut


def val():
    label = []
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint('./checkpoints/biLstm_crf')
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    content = sequence2id(pm.eva)
    pre_label = model.predict(session, content)
    label.extend(pre_label)

    return label


if __name__ == '__main__':
    pm = pm

    model = biLstm_crf()

    label = val()
    with open(pm.eva, 'r', encoding='utf-8') as f:
        sentences = [line.strip('\n') for line in f]

    for i in range(len(sentences)):
        sentence_cut = convert(sentences[i], label[i])
        print(sentences[i])
        print(sentence_cut)



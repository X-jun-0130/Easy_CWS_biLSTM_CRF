from Parameters import Parameters as pm
from data_processing import batch_iter, process
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode

class biLstm_crf(object):
    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, None], name='input_y')
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name='seq_length')
        self.keep_pro = tf.placeholder(tf.float32, name='drop_out')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.bils_crf()

    def bils_crf(self):

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.Variable(tf.truncated_normal([pm.vacab_size, pm.embedding_size], -0.25, 0.25), name='embedding')
            embeding_input = tf.nn.embedding_lookup(embedding, self.input_x)
            self.embedding = tf.nn.dropout(embeding_input, keep_prob=self.keep_pro)

        with tf.name_scope('Cell'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(pm.hidden_dim, state_is_tuple=True)
            Cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, self.keep_pro)

            cell_bw = tf.nn.rnn_cell.LSTMCell(pm.hidden_dim, state_is_tuple=True)
            Cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, self.keep_pro)

        with tf.name_scope('biLSTM'):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=Cell_fw, cell_bw=Cell_bw, inputs=self.embedding,
                                                         sequence_length=self.seq_length, dtype=tf.float32)
            outputs = tf.concat(outputs, 2)

        with tf.name_scope('output'):
            s = tf.shape(outputs)
            output = tf.reshape(outputs, [-1, 2*pm.hidden_dim])
            output = tf.layers.dense(output, pm.num_tags)
            output = tf.contrib.layers.dropout(output, self.keep_pro)
            self.logits = tf.reshape(output, [-1, s[1], pm.num_tags])

        with tf.name_scope('crf'):
            self.log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits, tag_indices=self.input_y,
                                                                             sequence_lengths=self.seq_length)
            # log_likelihood是对数似然函数，transition_params是转移概率矩阵
            # crf_log_likelihood{inputs:[batch_size,max_seq_length,num_tags],
            # tag_indices:[batchsize,max_seq_length],
            # sequence_lengths:[real_seq_length]
            # transition_params: A [num_tags, num_tags] transition matrix
            # log_likelihood: A scalar containing the log-likelihood of the given sequence of tag indices.

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(-self.log_likelihood)#最大似然取负，使用梯度下降

        with tf.name_scope('optimizer'):
            # 退化学习率 learning_rate = lr*(0.9**(global_step/10);staircase=True表示每decay_steps更新梯度
            # learning_rate = tf.train.exponential_decay(self.config.lr, global_step=self.global_step,
            # decay_steps=10, decay_rate=self.config.lr_decay, staircase=True)
            # optimizer = tf.train.AdamOptimizer(learning_rate)
            # self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step) #global_step 自动+1
            # no.2
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1

    def feed_data(self, x_batch, y_batch, seq_length, keep_pro):
        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.seq_length: seq_length,
                     self.keep_pro: keep_pro}
        return feed_dict

    def test(self, sess, x, y):
        batch_test = batch_iter(x, y, batch_size=pm.batch_size)
        for x_batch, y_batch in batch_test:
            x_batch, seq_length_x = process(x_batch)
            y_batch, seq_length_y = process(y_batch)
            feed_dict = self.feed_data(x_batch, y_batch, seq_length_x, 1.0)
            loss = sess.run(self.loss, feed_dict=feed_dict)
        return loss

    def predict(self, sess, x_batch):
        seq_pad, seq_length = process(x_batch)
        logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict={self.input_x: seq_pad,
                                                                                               self.seq_length: seq_length,
                                                                                               self.keep_pro: 1.0})
        label_ = []
        for logit, length in zip(logits, seq_length):
            # logit 每个子句的输出值，length子句的真实长度，logit[:length]的真实输出值
            # 调用维特比算法求最优标注序列
            viterbi_seq, _ = viterbi_decode(logit[:length], transition_params)
            label_.append(viterbi_seq)
        return label_






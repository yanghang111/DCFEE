# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn
# import rnncell as rnn


class Model(object):
    def __init__(self, config):
        self.num_filters = 200
        self.kernel_size = 5

        self.config = config
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.num_tags = 2
        self.num_chars = config["num_char"]

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model

        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")

        self.targets = tf.placeholder(dtype=tf.int64,
                                      shape=[None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        # embeddings for chinese character and segmentation representation
        lstm_inputs = self.embedding_layer(self.char_inputs,config)

        # bi-directional lstm layer
        lstm_output = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)

        ## CNN layer
        cnn_outputs = self.CNN_layer(lstm_inputs)

        # logits for tags
        self.logits = self.project_layer(cnn_outputs)

        # loss of the model
        self.loss = self.loss_layer(self.logits)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            # grads_vars = self.opt.compute_gradients(self.loss)
            # capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
            #                      for g, v in grads_vars]
            # self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        correct_prediction = tf.equal(tf.argmax(self.logits, -1), self.targets)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        # self.saver = tf.train.Saver(tf.global_variables(), reshape=True)

    def embedding_layer(self, char_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size],
        """
        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, 2*lstm_dim]
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            (outputs,
             (encoder_fw_final_state,
              encoder_bw_final_state))= tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
            final_state = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), -1)
        return final_state

    def CNN_layer(self, embedding_inputs):

        conv = tf.layers.conv1d(embedding_inputs, self.num_filters, self.kernel_size, name='conv')
        # global max pooling layer
        cnn_outputs = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
        return cnn_outputs

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        hidden_dim = self.lstm_dim * 2
        hidden_dim = int(self.num_filters/2)
        lstm_outputs = tf.reshape(lstm_outputs, [self.batch_size, hidden_dim])
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[hidden_dim, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                # output = tf.reshape(lstm_outputs, [self.batch_size, hidden_dim])
                hidden = tf.tanh(tf.nn.xw_plus_b(lstm_outputs, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_tags])

    def loss_layer(self, project_logits, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        # correct_prediction = tf.equal(tf.argmax(project_logits, 1), tf.argmax(self.targets, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        with tf.variable_scope("loss" if not name else name):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=project_logits, labels=self.targets))
            return loss

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        strings, chars, targets = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.dropout: 1.0,
        }
        feed_dict[self.targets] = np.asarray(targets)
        if is_train:
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            logits, acc = sess.run([self.logits, self.accuracy], feed_dict)
            return logits, acc

    def decode(self, logits):
        paths = []
        for score in (logits):
            path = tf.cast(tf.argmax(score, axis= -1), tf.int32).eval()
            paths.append(path)
        return paths

    def evaluate(self, sess, data_manager):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        for batch in data_manager.iter_batch():
            scores, acc = self.run_step(sess, False, batch)
            results.append(acc)
        acc_ = np.mean(results)
        return results, acc_

    def evaluate_line(self, sess, inputs):
        scores , acc = self.run_step(sess, False, inputs)
        x = self.decode(scores)
        pred = [int(x[0])]
        return pred

    def evaluete_(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        tp, tn, fp, fn = 0,0,0,0
        for batch in data_manager.iter_batch():
            actuals = tf.cast(batch[-1], tf.int64)
            scores, acc = self.run_step(sess, False, batch)
            predictions = tf.argmax(scores, 1)

            ones_like_actuals = tf.ones_like(actuals)
            zeros_like_actuals = tf.zeros_like(actuals)
            ones_like_predictions = tf.ones_like(predictions)
            zeros_like_predictions = tf.zeros_like(predictions)

            tp_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, ones_like_actuals),
                        tf.equal(predictions, ones_like_predictions)
                    ),
                    "float"
                )
            )

            tn_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, zeros_like_actuals),
                        tf.equal(predictions, zeros_like_predictions)
                    ),
                    "float"
                )
            )

            fp_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, zeros_like_actuals),
                        tf.equal(predictions, ones_like_predictions)
                    ),
                    "float"
                )
            )

            fn_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, ones_like_actuals),
                        tf.equal(predictions, zeros_like_predictions)
                    ),
                    "float"
                )
            )
            tp_, tn_, fp_, fn_ = sess.run([tp_op, tn_op, fp_op, fn_op])
            tp += tp_
            tn += tn_
            fp += fp_
            fn += fn_

        recall = float(tp) / (float(tp) + float(fn))
        precision = float(tp) / (float(tp) + float(fp))
        f1_score = (2 * (precision * recall)) / (precision + recall)
        # print("p,r,f1",precision, recall, f1_score)
        return precision, recall, f1_score
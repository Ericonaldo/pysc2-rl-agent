import os, argparse, pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from common import Config
from rl.agent import clip_log
from rl.model import fully_conv


# TODO extract this to an agent/ module
class ILAgent:
    def __init__(self, sess, model_fn, config, lr, clip_grads=1.):
        self.sess, self.config, self.lr = sess, config, lr
        (self.policy, self.value), self.inputs = model_fn(config)
        self.actions = [tf.placeholder(tf.int32, [None]) for _ in self.policy]

        loss_fn, self.loss_inputs = self._loss_func()

        self.step = tf.Variable(0, trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)
        # opt = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99, epsilon=1e-5)
        self.train_op = layers.optimize_loss(loss=loss_fn, optimizer=opt, learning_rate=None, global_step=self.step, clip_gradients=clip_grads)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('logs/' + self.config.full_id(), graph=None)

    def train(self, states, actions):
        feed_dict = {self.inputs: states, self.actions: actions}
        result, result_summary, step = self.sess.run([self.train_op, self.summary_op, self.step], feed_dict)

        self.summary_writer.add_summary(result_summary, step)

        return result

    def _loss_func(self):
        acts = [tf.one_hot(self.actions[i], d) for i, (d, _) in enumerate(self.config.policy_dims())]
        ce = sum([-tf.reduce_sum(a * clip_log(p), axis=-1) for a, p in zip(acts, self.policy)])
        ce_loss = tf.reduce_mean(ce)
        val_loss = 0 * tf.reduce_mean(self.value) # hack to match a2c agent computational graph
        return ce_loss + val_loss, actions

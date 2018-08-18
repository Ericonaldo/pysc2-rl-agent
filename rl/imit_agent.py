# -*- coding: utf-8 -*-
import os, argparse, pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from common import Config
from rl.agent import clip_log
from rl.model import fully_conv
from pysc2.lib.actions import FUNCTIONS

FUNCTION_LIST = np.array(FUNCTIONS)

# TODO extract this to an agent/ module
class ILAgent:
    def __init__(self, sess, model_fn, config, lr, restore=False, clip_grads=1.):
        self.sess, self.config, self.lr = sess, config, lr
        (self.policy, self.value), self.inputs = model_fn(config) # self.inputs = [screen_input, minimap_input] + non_spatial_inputs
        self.actions = [tf.placeholder(tf.int32, [None]) for _ in range(len(self.policy))] # policy is a list, actions is a list  每个元素对应动作函数或参数
        #print(self.inputs)
        #print(self.actions)

        with tf.variable_scope('loss'):
            acts=[]
            for i, (d, is_spatial) in enumerate(self.config.policy_dims()):
                if is_spatial:
                    acts.append(tf.one_hot(self.actions[i], config.sz * config.sz))
                else:
                    acts.append(tf.one_hot(self.actions[i], d))
            # acts = self.mask(self.actions[0], acts) # TODO
            ce = sum([-tf.reduce_sum(a * clip_log(p), axis=-1) for a, p in zip(acts, self.policy)])
            ce_loss = tf.reduce_mean(ce)
            val_loss = 0 * tf.reduce_mean(self.value) # hack to match a2c agent computational graph
            self.loss = ce_loss + val_loss
            tf.summary.scalar('loss', self.loss)
        
        with tf.variable_scope('train'):
            self.step = tf.Variable(0, trainable=False)
            # opt = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)
            opt = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99, epsilon=1e-5)
            self.train_op = layers.optimize_loss(loss=self.loss, optimizer=opt, learning_rate=None, global_step=self.step, clip_gradients=clip_grads)
            self.sess.run(tf.global_variables_initializer())     

        self.saver = tf.train.Saver()
        if restore:
            self.saver.restore(self.sess, tf.train.latest_checkpoint('weights/' + self.config.full_id()))

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('supervised_logs/' + self.config.full_id(), graph=None)

    def train(self, states, actions):
        # print(len(states))
        # for i in states:
        #     print(i.shape)
        # print(len(actions))
        # feed_dict = {self.inputs: states, self.actions: actions}
        feed_dict = dict(zip(self.inputs + self.actions, states + actions))
        result, result_summary, step = self.sess.run([self.train_op, self.summary_op, self.step], feed_dict)

        self.summary_writer.add_summary(result_summary, step)

        return result
    
    # TODO
    '''
    def mask(self, action_nums, acts):
        param_list = [[self.config.arg_idx[FUNCTION_LIST[np.array(self.actions[0])].args[i].name]+1] for i in range(FUNCTION_LIST[np.array(self.actions[0])].args)]
        print(param_list)
        for i in range(action_nums):
            for j, (d, is_spatial) in enumerate(self.config.policy_dims()):
                if j not in params[i]:
                    acts[i] = acts[i] * np.zeros(acts[i].shape)
    '''
        

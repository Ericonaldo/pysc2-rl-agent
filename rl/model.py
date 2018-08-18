# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import layers
from pysc2.lib import actions
import numpy as np

# Functions
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id

FUNCTION_LIST=[_NOOP, _SELECT_POINT, _MOVE_SCREEN]
NO_FUNCTION_LIST=[i for i in range(len(actions.FUNCTIONS)) if i not in FUNCTION_LIST]
# 定义和实现了模型

def fully_conv(config):
    screen, screen_input = cnn_block(config.sz, config.screen_dims(), config.embed_dim_fn) # [None, 32, 32, 32]
    minimap, minimap_input = cnn_block(config.sz, config.minimap_dims(), config.embed_dim_fn) # [None, 32, 32, 32]
    non_spatial, non_spatial_inputs = non_spatial_block(config.sz, config.non_spatial_dims(), config.ns_idx) # [None, 11, 32, 32]

    state = tf.concat([screen, minimap, non_spatial], axis=1) # [None, 64+11, 32, 32]
    fc1 = layers.fully_connected(layers.flatten(state), num_outputs=256) # [None, 64*32*32]->[None, 256]
    value = tf.squeeze(layers.fully_connected(fc1, num_outputs=1, activation_fn=None), axis=1) # [None, 1]->[None]

    # TODO mask unused args
    # inspired by https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/networks/fully_conv.py#L131-L137
    policy = []
    for dim, is_spatial in config.policy_dims(): # [(524, 0), (0, True), (0, True), (0, True), (2, False), (5, False), (10, False), (4, False), (2, False), (4, False), (500, False), (4, False), (10, False), (500, False)]
        if is_spatial:
            logits = layers.conv2d(state, num_outputs=1, kernel_size=1, activation_fn=None, data_format="NCHW") # [None, 1, 32, 32]
            policy.append(tf.nn.softmax(layers.flatten(logits))) # [None, 32*32=1024]
        else:
            policy.append(layers.fully_connected(fc1, num_outputs=dim, activation_fn=tf.nn.softmax)) # [None, dim]
    policy[0] = mask_probs(policy[0], non_spatial_inputs[config.ns_idx['available_actions']], config.restrict) 

    return [policy, value], [screen_input, minimap_input] + non_spatial_inputs # policy[0]是one_hot动作函数，[None, 524];  policy[1:]为13个one_hot参数[[None, dim1], [None, dim2],...], spatial的feature维度是1024


def cnn_block(sz, dims, embed_dim_fn): # CNN(空间信息层)
    block_input = tf.placeholder(tf.float32, [None, sz, sz, len(dims)]) # dims 为列表，用了几个feature map则len(dims)=几, sz默认为32 　->　 [None, 32, 32, len(dims)]
    block = tf.transpose(block_input, [0, 3, 1, 2]) # NHWC -> NCHW   [None, len(dims), 32, 32]

    block = tf.split(block, len(dims), axis=1)  # [[None, 1, 32, 32], [None, 1, 32, 32], [...]]
    for i, d in enumerate(dims): # dims = [dim_subinfo1, dim_subinfo2, ...]
        if d > 1:
            block[i] = tf.one_hot(tf.to_int32(tf.squeeze(block[i], axis=1)), d, axis=1) # block[i] = [None, 32, 32], block[i] = [None, one_hot_num, 32, 32]
            block[i] = layers.conv2d(block[i], num_outputs=embed_dim_fn(d), kernel_size=1, data_format="NCHW") # block[i] = [None, num_outputs, 32, 32]
        else:
            block[i] = tf.log(block[i] + 1.0) # [None, 1, 32, 32]
    block = tf.concat(block, axis=1) # [None, sum_num_outputs, 32, 32]

    conv1 = layers.conv2d(block, num_outputs=16, kernel_size=5, data_format="NCHW") # [None, 16, 32, 32], 默认0填充边缘
    conv2 = layers.conv2d(conv1, num_outputs=32, kernel_size=3, data_format="NCHW") # [None, 32, 32, 32]

    return conv2, block_input


def non_spatial_block(sz, dims, idx): # 非空间信息层
    block_inputs = [tf.placeholder(tf.float32, [None, *dim]) for dim in dims] # *dim指任意多的参数，因此dim可以是数字也可以是tuple，list，[[None, *dim1], [None, *dim2], ...]
    # TODO currently too slow with full inputs
    # block = [broadcast(block_inputs[i], sz) for i in range(len(dims))]
    # block = tf.concat(block, axis=1)
    block = broadcast(tf.log(block_inputs[idx['player']] + 1.0), sz) # block_inputs[idx['player']] = [None, 11] -> [None, 11, 32, 32]
    return block, block_inputs # 仅使用了player信息，available action作为参考


# based on https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/networks/fully_conv.py#L91-L96
def broadcast(tensor, sz):
    return tf.tile(tf.expand_dims(tf.expand_dims(tensor, 2), 3), [1, 1, sz, sz]) # 扩展维度为 [x, x, sz, sz]


def mask_probs(probs, mask, restrict=False): # 将 available_actions 中不允许的动作概率置为0, restrict表示是否限制动作输出
    masked = probs * mask # [None, 524]
    function_mask = np.ones(masked.shape[1])
    if restrict:
        function_mask = np.zeros(masked.shape[1])
        function_mask[FUNCTION_LIST] = 1
    
    # print('------',function_mask)
    function_mask = tf.constant(function_mask)
    function_mask = tf.cast(tf.expand_dims(function_mask, 0), dtype=masked.dtype)
    masked *= function_mask

    correction = tf.cast(
        tf.reduce_sum(masked, axis=-1, keep_dims=True) < 1e-3, dtype=tf.float32
        ) * (1.0 / (tf.reduce_sum(mask, axis=-1, keep_dims=True) + 1e-12)) * mask * function_mask  # 第一项判断是否小于阈值， 第二项乘 (1/可用动作数量)
    masked += correction
    return masked / tf.clip_by_value(tf.reduce_sum(masked, axis=1, keep_dims=True), 1e-12, 1.0) # 归一化

# -*- coding: utf-8 -*-
import sys, os, argparse, pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from common import Config
from rl.agent import clip_log
from rl.model import fully_conv
from rl.imit_agent import *
from common.dataset import *

from pysc2.lib.actions import FUNCTIONS
FUNCTION_LIST = np.array(FUNCTIONS)

# TODO extract this to an agent/ module
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sz", type=int, default=32)
    parser.add_argument('--lr', type=float, default=7e-4)
    # parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_sz', type=int, default=256)
    parser.add_argument("--map", type=str, default='MoveToBeacon')
    parser.add_argument("--restore", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--cfg_path", type=str, default='config.json.dist')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tf.reset_default_graph()
    sess = tf.Session()

    config = Config(args.sz, args.map, -1, imitation=True)
    os.makedirs('weights/' + config.full_id() + '_imitation', exist_ok=True)
    cfg_path = 'weights/%s_imitation/config.json' % config.full_id()
    config.build(args.cfg_path)
    config.save(cfg_path)
    
    dataset = Dataset()
    dataset.load("replay/"+config.full_id())

    inputs = config.preprocess(dataset.input_observations)
    outputs = dataset.output_actions.transpose()
    
    # args = FUNCTION_LIST[outputs[0]][1][5]
    temp_param_list = [action[5] for action in FUNCTION_LIST[outputs[0]]] # 每个sample的function所具有的参数[[fuc1.arg1,fuc1.arg2], [fuc2.arg1], ...]
    param_list = []
    for sam in temp_param_list:
        sam = [config.arg_idx[i.name] for i in sam] 
        param_list.append(sam)
    # param_list = np.array(param_list) #转化为参数id[[fuc1.arg1.id,fuc1.arg2.id], [fuc2.arg1.id], ...]
    
    d = config.policy_dims() # 提取参数mask以在模型中免去计算不存在参数的loss
    param_one_hot = []
    dim = d[0][0]
    param_one_hot.append(np.ones([len(outputs[0]), dim])) # act function全为1
    d = d[1:]
    for i in range(len(d)):
        if d[i][1]: # if is spatial
            dim = config.sz * config.sz
        else:
            dim = d[i][0]
        param = np.zeros([len(outputs[0]), dim])
        for j in range(param.shape[0]):
            if i in param_list[j]:
                param[j,:]=1
        # print(i, param)
        param_one_hot.append(param)
    # print(len(param_one_hot)) # param one-hot mask
    
    rollouts = [inputs, outputs, param_one_hot]

    agent = ILAgent(sess, fully_conv, config, args.lr, args.restore)

    # for i in rollouts[0]:
    #    print(np.shape(i))

    n_samples = len(rollouts[0][0])
    epochs = args.epochs
    # epochs = max(1, samples // n)
    n_batches = n_samples // args.batch_sz + 1
    print("n_samples: %d, epochs: %d, batches: %d" % (n_samples, epochs, n_batches))
    for e in range(epochs):
        print("epoch {} begin".format(e))
        for _ in range(n_batches):
            idx = np.random.choice(n_samples, args.batch_sz, replace=False)
            sample = [s[idx] for s in rollouts[0]], [a[idx] for a in rollouts[1]], [p[idx] for p in rollouts[2]]
            # print(sample)
            res = agent.train(*sample)
            # print(res)
        print("after this epoch, loss is ", res)
        agent.saver.save(sess, 'weights/%s/a2c_%d' % (config.full_id(), e), global_step=agent.step)

# -*- coding: utf-8 -*-
import os, argparse
from absl import flags
import tensorflow as tf

from common import Config
from common.env import make_envs
from rl.agent import A2CAgent
from rl.model import fully_conv
from rl import Runner, EnvWrapper

if __name__ == '__main__':
    flags.FLAGS(['main.py']) # 注册Flag对象，这里应该没有用
    '''解析命令行参数'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sz", type=int, default=32)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--updates", type=int, default=1000000)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--vf_coef', type=float, default=0.25)
    parser.add_argument('--ent_coef', type=float, default=1e-3)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--clip_grads', type=float, default=1.)
    parser.add_argument("--run_id", type=int, default=-1)
    parser.add_argument("--map", type=str, default='MoveToBeacon')
    parser.add_argument("--cfg_path", type=str, default='config.json.dist')
    parser.add_argument("--restrict", type=bool, default=False)
    parser.add_argument("--imitation", type=bool, default=False)
    parser.add_argument("--test", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--restore", type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--save_replay', type=bool, nargs='?', const=True, default=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) #environ是一个字符串所对应环境的映像对象，输入为要使用的GPU number
    tf.reset_default_graph()
    sess = tf.Session()

    # config = Config(args.sz, args.map, lambda _: 1)
    config = Config(args.sz, args.map, args.run_id,restrict=args.restrict,imitation=args.imitation) # 进行参数的设置
    os.makedirs('weights/' + config.full_id(), exist_ok=True)
    cfg_path = 'weights/%s/config.json' % config.full_id() # 保存参数的位置
    config.build(cfg_path if args.restore else args.cfg_path) # 建立和设置参数
    if not args.restore and not args.test:
        config.save(cfg_path) # 保存参数

    envs = EnvWrapper(make_envs(args), config) # 创建环境,封装一层
    agent = A2CAgent(sess, fully_conv, config, args.restore, args.discount, args.lr, args.vf_coef, args.ent_coef, args.clip_grads) # 创建agent

    runner = Runner(envs, agent, args.steps) # 创建进程
    runner.run(args.updates, not args.test) # 开始运行

    if args.save_replay: #是否保存回放
        envs.save_replay()

    envs.close() # 关闭环境

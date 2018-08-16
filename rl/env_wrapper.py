# -*- coding: utf-8 -*-
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from common.config import DEFAULT_ARGS, is_spatial

# 环境的包装，将动作包装为环境需要的格式， 将返回的数据也包装为可用的格式


class EnvWrapper:
    def __init__(self, envs, config):
        self.envs, self.config = envs, config

    def step(self, acts):
        acts = self.wrap_actions(acts) # 包装acts
        results = self.envs.step(acts) # 包装results
        return self.wrap_results(results)

    def reset(self):
        results = self.envs.reset()
        return self.wrap_results(results)

    def wrap_actions(self, actions):
        acts, args = actions[0], actions[1:]

        wrapped_actions = []
        for i, act in enumerate(acts): # 对于batch中的每个act函数
            act_args = []
            for arg_type in FUNCTIONS[act].args: # 对于该动作函数的每个参数
                act_arg = [DEFAULT_ARGS[arg_type.name]] # 初始化
                if arg_type.name in self.config.act_args:
                    act_arg = [args[self.config.arg_idx[arg_type.name]][i]] # 等于bacth中对应参数的第i项(第i个样本)
                if is_spatial(arg_type.name):  # spatial args, convert to coords
                    act_arg = [act_arg[0] % self.config.sz, act_arg[0] // self.config.sz]  # (y,x), fix for PySC2
                act_args.append(act_arg)
            wrapped_actions.append(FunctionCall(act, act_args))

        return wrapped_actions

    def wrap_results(self, results): # results 由每个进程组成的list
        obs = [res.observation for res in results] 
        rewards = [res.reward for res in results]
        dones = [res.last() for res in results]

        states = self.config.preprocess(obs)

        return states, rewards, dones

    def save_replay(self, replay_dir='PySC2Replays'):
        self.envs.save_replay(replay_dir)

    def spec(self):
        return self.envs.spec()

    def close(self):
        return self.envs.close()

    @property
    def num_envs(self):
        return self.envs.num_envs

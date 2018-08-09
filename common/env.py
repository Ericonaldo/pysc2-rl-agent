from pysc2.env import sc2_env
from multiprocessing import Process, Pipe
^
|
|
# 多进程及通信

def make_envs(args):
    env_args = dict(map_name=args.map, step_mul=8, game_steps_per_episode=0) 
    return EnvPool([make_env(args.sz, **dict(env_args, visualize=i < args.render)) for i in range(args.envs)]) # envs要创建的表示进程数量，每个进程创建一个环境，每个环境由i<render来决定是否显示


# based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(sz=32, **params):
'''加入两个参数并转换为SC2_ENV环境'''
    def _thunk():
        params['screen_size_px'] = params['minimap_size_px'] = (sz, sz)
        env = sc2_env.SC2Env(**params)
        return env
    return _thunk


# based on https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
# SC2Env::step expects actions list and returns obs list so we send [data] and process obs[0]
def worker(remote, env_fn_wrapper): # 接收进程发来的命令并执行并返回执行内容
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'spec':
            remote.send((env.observation_spec(), env.action_spec()))
        elif cmd == 'step':
            obs = env.step([data])
            remote.send(obs[0])
        elif cmd == 'reset':
            obs = env.reset()
            remote.send(obs[0])
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'save_replay':
            env.save_replay(data)
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class EnvPool(object): # 进程池
    def __init__(self, env_fns):
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)]) # 创建多个通信通道，Pipe()得到pipe[0]和pipe[1]分别表示发送和接收
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn))) # 设置多进程并开启
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

    def spec(self): # 进程池接收每个进程的observation
        for remote in self.remotes:
            remote.send(('spec', None))
        results = [remote.recv() for remote in self.remotes]
        # todo maybe support running different envs / specs in the future?
        return results[0]

    def step(self, actions): # 进程池（同步）执行每个进程
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        return results

    def reset(self): # 进程池初始化所有进程
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def close(self): # 进程池关闭所有进程
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def save_replay(self, replay_dir='PySC2Replays'): # 保存第0个进程的replay
        self.remotes[0].send(('save_replay', replay_dir))

    @property
    def num_envs(self): # 返回进程的数量
        return len(self.remotes)

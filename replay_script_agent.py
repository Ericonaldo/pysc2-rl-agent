# -*- coding: utf-8 -*-
# __author__ = 'Tony Beltramelli - www.tonybeltramelli.com' + 'inorry'
# __modify__ = 'eric'
# scripted agents taken from PySC2, credits to DeepMind
# https://github.com/deepmind/pysc2/blob/master/pysc2/agents/scripted_agent.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from future.builtins import range  # pylint: disable=redefined-builtin

import numpy as np
import pickle
import uuid
import sys
import os
import importlib
import threading
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch

from absl import app
from absl import flags
import copy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib.actions import FUNCTIONS
from pysc2.lib import features

from common.config import *

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = 0
_SELECT_ALL = 0

DATA_SIZE = 10

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("sz", 32,
                     "Resolutions.")
flags.DEFINE_integer("screen_resolution", 84,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_agent_steps", DATA_SIZE+1, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(),
                  "Bot's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use.")
flags.mark_flag_as_required("map")

class MoveToBeacon(base_agent.BaseAgent):
  """An agent specifically for solving the MoveToBeacon map."""
  def __init__(self, config):
    base_agent.BaseAgent.__init__(self)
    self.name = self.__class__.__name__
    self.states = []
    self.action = 0
    self.param = []
    self.config = config
    for arg in actions.TYPES._fields:
       self.param.append(DEFAULT_ARGS[arg])  
  def step(self, obs):
    super(MoveToBeacon, self).step(obs)
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      if not neutral_y.any():
        self.action = _NO_OP # 动作函数id
        param = []
      target_xy = [int(neutral_x.mean()), int(neutral_y.mean())]
      target = int(neutral_x.mean()) * self.config.sz + int(neutral_y.mean())
      self.action = _MOVE_SCREEN # 动作函数id
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[0].name]] = _NOT_QUEUED
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[1].name]] = target # 函数参数
      param = [[_NOT_QUEUED], target_xy]
    else:
      self.action = _SELECT_ARMY # 动作函数id
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[0].name]] = _SELECT_ALL # 函数参数
      param = [[_SELECT_ALL]]
            
    # self.states.append(np.array([obs.observation, self.action, self.param]))
    
    self.states.append([copy.deepcopy(obs.observation), copy.deepcopy(self.action), copy.deepcopy(self.param)])

    if len(self.states) == DATA_SIZE:
      new_file_name = str(uuid.uuid1())
      # np.save('replay/' + config.full_id() +'/{}'.format(new_file_name), np.array(self.states))
      pickle.dump(self.states, open('replay/' + config.full_id() +'/{}'.format(new_file_name) + '_{}.replay'.format(DATA_SIZE) ,'wb'))
      self.states = []
      
    return actions.FunctionCall(self.action, param)


class CollectMineralShards(base_agent.BaseAgent):
  """An agent specifically for solving the CollectMineralShards map."""
  def __init__(self, config):
    base_agent.BaseAgent.__init__(self)
    self.name = self.__class__.__name__
    self.states = []
    self.action = 0
    self.param = []
    self.config = config
    for arg in actions.TYPES._fields:
       self.param.append(DEFAULT_ARGS[arg])
  def step(self, obs):
    super(CollectMineralShards, self).step(obs)
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      if not neutral_y.any() or not player_y.any():
        self.action = _NO_OP # 动作函数id
        param = []
      player = [int(player_x.mean()), int(player_y.mean())]
      # player = int(player_x.mean()) * self.config.sz + int(player_y.mean())
      closest, min_dist = None, None
      for p in zip(neutral_x, neutral_y):
        dist = np.linalg.norm(np.array(player) - np.array(p))
        if not min_dist or dist < min_dist:
          closest, min_dist = p, dist
      self.action = _MOVE_SCREEN # 动作函数id
      param = [[_NOT_QUEUED], closest]
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[0].name]] = _NOT_QUEUED
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[1].name]] = closest[0] * self.config.sz +closest[1]
    else:
      self.action = _SELECT_ARMY # 动作函数id
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[0].name]] = _SELECT_ALL
      param = [[_SELECT_ALL]]

    self.states.append([copy.deepcopy(obs.observation), copy.deepcopy(self.action), copy.deepcopy(self.param)])

    if len(self.states) == DATA_SIZE:
      new_file_name = str(uuid.uuid1())
      # np.save('replay/' + config.full_id() +'/{}'.format(new_file_name), np.array(self.states))
      pickle.dump(self.states, open('replay/' + config.full_id() +'/{}'.format(new_file_name) + '_{}.replay'.format(DATA_SIZE) ,'wb'))
      self.states = []
      
    return actions.FunctionCall(self.action, param)


class DefeatRoaches(base_agent.BaseAgent):
  """An agent specifically for solving the DefeatRoaches map."""
  def __init__(self, config):
    base_agent.BaseAgent.__init__(self)
    self.name = self.__class__.__name__
    self.states = []
    self.action = 0
    self.param = []
    self.config = config
    for arg in actions.TYPES._fields:
       self.param.append(DEFAULT_ARGS[arg])  
  def step(self, obs):
    super(DefeatRoaches, self).step(obs)
    if _ATTACK_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      roach_y, roach_x = (player_relative == _PLAYER_HOSTILE).nonzero()
      if not roach_y.any():
        self.action = _NO_OP
        param = []
      index = np.argmax(roach_y)
      target_xy = [roach_x[index], roach_y[index]]
      target = roach_x[index]*self.config.sz + roach_y[index]
      self.action = _ATTACK_SCREEN
      param = [[_NOT_QUEUED], target_xy]
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[0].name]] = _NOT_QUEUED
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[1].name]] = target # 函数参数
    elif _SELECT_ARMY in obs.observation["available_actions"]:
      self.action = _SELECT_ARMY
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[0].name]] = _SELECT_ALL
      param = [[_SELECT_ALL]]
    else:
      self.action = _NO_OP,
      param = []
      
    
    self.states.append([copy.deepcopy(obs.observation), copy.deepcopy(self.action), copy.deepcopy(self.param)])

    if len(self.states) == DATA_SIZE:
      new_file_name = str(uuid.uuid1())
      # np.save('replay/' + config.full_id() +'/{}'.format(new_file_name), np.array(self.states))
      pickle.dump(self.states, open('replay/' + config.full_id() +'/{}'.format(new_file_name) + '_{}.replay'.format(DATA_SIZE) ,'wb'))
      self.states = []
      
    return actions.FunctionCall(self.action, param)
    

def run_thread(agent_cls, map_name, visualize):
    with sc2_env.SC2Env(
        map_name=map_name,
        agent_race=FLAGS.agent_race,
        bot_race=FLAGS.bot_race,
        difficulty=FLAGS.difficulty,
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=FLAGS.game_steps_per_episode,
        screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
        minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
        visualize=visualize) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        agent = agent_cls
        run_loop.run_loop([agent], env, FLAGS.max_agent_steps)
        #if FLAGS.save_replay:
        #    env.save_replay(agent_cls.name)

if __name__ == "__main__":
    """Run an agent."""
    FLAGS(sys.argv)
    
    print("-------------------")
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace
    
    maps.get(FLAGS.map)  # Assert the map exists.
    FLAGS.screen_resolution = FLAGS.minimap_resolution = FLAGS.sz # 强制让screen和minimap的大小都为sz

    config = Config(FLAGS.sz, FLAGS.map, -1) # 进行参数的设置
    os.makedirs('script_weights/' + config.full_id(), exist_ok=True)
    os.makedirs('replay/' + config.full_id(), exist_ok=True)
    cfg_path = 'script_weights/%s/config.json' % config.full_id() # 保存参数的位置
    config.build('config.json.dist') # 建立和设置参数
    config.save(cfg_path)

    #agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    #agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    if FLAGS.map == "CollectMineralShards":
        agent_cls = CollectMineralShards(config)
    elif FLAGS.map == "MoveToBeacon":
        agent_cls = MoveToBeacon(config)
    elif FLAGS.map == "DefeatRoaches":    
        agent_cls = DefeatRoaches(config)

    threads = []
    for _ in range(FLAGS.parallel - 1):
        t = threading.Thread(target=run_thread, args=(agent_cls, FLAGS.map, False))
        threads.append(t)
        t.start()

    run_thread(agent_cls, FLAGS.map, FLAGS.render)

    for t in threads:
        t.join()

    if FLAGS.profile:
        print(stopwatch.sw)

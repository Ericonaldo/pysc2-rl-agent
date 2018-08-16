__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'
__modify__ = 'eric'
# scripted agents taken from PySC2, credits to DeepMind
# https://github.com/deepmind/pysc2/blob/master/pysc2/agents/scripted_agent.py

import numpy as np
import uuid

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib.actions import FUNCTIONS
from pysc2.lib import features

from Utils import *
from common.config import DEFAULT_ARGS, is_spatial, SZ

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

DATA_SIZE = 100000

class MoveToBeacon(base_agent.BaseAgent):
  """An agent specifically for solving the MoveToBeacon map."""
  def __init__(self):
    base_agent.BaseAgent.__init__(self)
    self.states = []
    self.action = 0
    self.param = []
    for arg in actions.TYPES._fields:
       self.param.append([DEFAULT_ARGS[arg]])  
  def step(self, obs):
    super(MoveToBeacon, self).step(obs)
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      if not neutral_y.any():
        self.action = _NO_OP # 动作函数id
        # param = []
      # target = [int(neutral_x.mean()), int(neutral_y.mean())]
      target = int(neutral_x.mean()) * self.config.SZ + int(neutral_y.mean())
      self.action = _MOVE_SCREEN # 动作函数id
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[0].name]] = [_NOT_QUEUED]
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[1].name]] = [target] # 函数参数
      # param = [_NOT_QUEUED, target]
    else:
      self.action = _SELECT_ARMY # 动作函数id
      self.param[self.config.arg_idx[FUNCTIONS[act].args[0].name]] = [_SELECT_ALL] # 函数参数
      # param = [_SELECT_ALL]
            
    self.states.append(np.array([obs.observation, self.action, self.param]))

    if len(self.states) == DATA_SIZE:
      new_file_name = str(uuid.uuid1())
      np.save("dataset_{}/{}".format(self.__class__.__name__, new_file_name), np.array(self.states))
      self.states = []
      
    return actions.FunctionCall(self.action, self.param)


class CollectMineralShards(base_agent.BaseAgent):
  """An agent specifically for solving the CollectMineralShards map."""
  def __init__(self):
    base_agent.BaseAgent.__init__(self)
    self.states = []
    self.action = 0
    self.param = []
    for arg in actions.TYPES._fields:
       self.param.append([DEFAULT_ARGS[arg]])   
  def step(self, obs):
    super(CollectMineralShards, self).step(obs)
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      if not neutral_y.any() or not player_y.any():
        self.action = _NO_OP # 动作函数id
        # self.param = []
      player = [int(player_x.mean()), int(player_y.mean())]
      # player = int(player_x.mean()) * self.config.SZ + int(player_y.mean())
      closest, min_dist = None, None
      for p in zip(neutral_x, neutral_y):
        dist = np.linalg.norm(np.array(player) - np.array(p))
        if not min_dist or dist < min_dist:
          closest, min_dist = p, dist
      self.action = _MOVE_SCREEN # 动作函数id
      # param = [_NOT_QUEUED, closest]
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[0].name]] = [_NOT_QUEUED]
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[1].name]] = [closest[0] * config.SZ +closest[1]]
    else:
      self.action = _SELECT_ARMY # 动作函数id
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[0].name]] = [_SELECT_ALL]
      # param = [_SELECT_ALL]
      
    self.states.append(np.array([obs.observation, self.action, self.param]))

    if len(self.states) == DATA_SIZE:
      new_file_name = str(uuid.uuid1())
      np.save("dataset_{}/{}".format(self.__class__.__name__, new_file_name), np.array(self.states))
      self.states = []
      
    return actions.FunctionCall(self.action, self.param)


class DefeatRoaches(base_agent.BaseAgent):
  """An agent specifically for solving the DefeatRoaches map."""
  def __init__(self):
    base_agent.BaseAgent.__init__(self)
    self.name = self.__class__.__name__
    self.states = []
    self.action = 0
    self.param = []
    for arg in actions.TYPES._fields:
       self.param.append([DEFAULT_ARGS[arg]])  
  def step(self, obs):
    super(DefeatRoaches, self).step(obs)
    if _ATTACK_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      roach_y, roach_x = (player_relative == _PLAYER_HOSTILE).nonzero()
      if not roach_y.any():
        self.action = _NO_OP
        # param = []
      index = np.argmax(roach_y)
      # target = [roach_x[index], roach_y[index]]
      target = roach_x[index]*SZ + roach_y[index]
      self.action = _ATTACK_SCREEN
      # param [_NOT_QUEUED, target]
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[0].name]] = [_NOT_QUEUED]
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[1].name]] = [target] # 函数参数
    elif _SELECT_ARMY in obs.observation["available_actions"]:
      self.action = _SELECT_ARMY
      self.param[self.config.arg_idx[FUNCTIONS[self.action].args[0].name]] = [_SELECT_ALL]
      # param = [_SELECT_ALL]
    else:
      self.action = _NO_OP,
      # param = []
      
    self.states.append(np.array([obs.observation, self.action, self.param]))

    if len(self.states) == DATA_SIZE:
      new_file_name = str(uuid.uuid1())
      np.save("dataset_{}/{}".format(self.__class__.__name__, new_file_name), np.array(self.states))
      self.states = []
      
    return actions.FunctionCall(self.action, self.param)

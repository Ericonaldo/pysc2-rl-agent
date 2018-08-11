__author__ = 'eric'
# scripted agents taken from PySC2, credits to DeepMind
# https://github.com/deepmind/pysc2/blob/master/pysc2/agents/scripted_agent.py

import numpy
import uuid

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from Utils import *

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

class MoveToBeacon(base_agent.BaseAgent):
  """An agent specifically for solving the MoveToBeacon map."""
  def __init__(self):
    base_agent.BaseAgent.__init__(self)
    self.states = []
  def step(self, obs):
    super(MoveToBeacon, self).step(obs)
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      if not neutral_y.any():
        action = _NO_OP
        param = []
      target = [int(neutral_x.mean()), int(neutral_y.mean())]
      action = _MOVE_SCREEN
      param = [_NOT_QUEUED, target]
    else:
      action = _SELECT_ARMY
      param = [_SELECT_ALL]
      
    self.states.append(np.array([obs.observation, action, params]))

    if len(self.states) == 64:
      new_file_name = str(uuid.uuid1())
      np.save("dataset_{}/{}".format(name, new_file_name), np.array(self.states))
      self.states = []
      
    return actions.FunctionCall(action, param)


class CollectMineralShards(base_agent.BaseAgent):
  """An agent specifically for solving the CollectMineralShards map."""
  def __init__(self):
    base_agent.BaseAgent.__init__(self)
    self.states = []
  def step(self, obs):
    super(CollectMineralShards, self).step(obs)
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      if not neutral_y.any() or not player_y.any():
        action = _NO_OP
        param = []
      player = [int(player_x.mean()), int(player_y.mean())]
      closest, min_dist = None, None
      for p in zip(neutral_x, neutral_y):
        dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
        if not min_dist or dist < min_dist:
          closest, min_dist = p, dist
      action = _MOVE_SCREEN
      param = [_NOT_QUEUED, closest]
    else:
      action = _SELECT_ARMY
      param = [_SELECT_ALL]
      
    self.states.append(np.array([obs.observation, action, params]))

    if len(self.states) == 64:
      new_file_name = str(uuid.uuid1())
      np.save("dataset_{}/{}".format(name, new_file_name), np.array(self.states))
      self.states = []
      
    return actions.FunctionCall(action, param)


class DefeatRoaches(base_agent.BaseAgent):
  """An agent specifically for solving the DefeatRoaches map."""
  def __init__(self):
    base_agent.BaseAgent.__init__(self)
    self.name = self.__class__.__name__
    self.states = []
  def step(self, obs):
    super(DefeatRoaches, self).step(obs)
    if _ATTACK_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      roach_y, roach_x = (player_relative == _PLAYER_HOSTILE).nonzero()
      if not roach_y.any():
        action = _NO_OP
        param = []
      index = numpy.argmax(roach_y)
      target = [roach_x[index], roach_y[index]]
      action = _ATTACK_SCREEN
      param [_NOT_QUEUED, target]
    elif _SELECT_ARMY in obs.observation["available_actions"]:
      action = _SELECT_ARMY
      param = [_SELECT_ALL]
    else:
      action = _NO_OP,
      param = []
      
    self.states.append(np.array([obs.observation, action, params]))

    if len(self.states) == 64:
      new_file_name = str(uuid.uuid1())
      np.save("dataset_{}/{}".format(name, new_file_name), np.array(self.states))
      self.states = []
      
    return actions.FunctionCall(action, param)


class ScriptedAgent(base_agent.BaseAgent):
    def __init__(self):
        base_agent.BaseAgent.__init__(self)

        self.states = []

    def step(self, obs):
        super(ScriptedAgent, self).step(obs)

        observation = obs.observation["minimap"][5]
        observation = Utils.feature_array_to_img(observation, max_target_value=1.0)
        observation = Utils.resize_squared_img(observation, 84)
        #Utils.show(observation)

        if GAME == "beacon":
            if actions.FUNCTIONS.Move_screen.id in obs.observation["available_actions"]:
                player_relative = obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index]
                neutral_y, neutral_x = (player_relative == 3).nonzero()

                if not neutral_y.any():
                    action = actions.FUNCTIONS.no_op.id
                    params = []
                else:
                    target = [int(neutral_x.mean()), int(neutral_y.mean())]

                    action = actions.FUNCTIONS.Move_screen.id
                    params = [[0], target] # _SELECT_ALL = [0]
            else:
                action = actions.FUNCTIONS.select_army.id
                params = [[0]] # _SELECT_ALL = [0]
        elif GAME == "mineral":
            if actions.FUNCTIONS.Move_screen.id in obs.observation["available_actions"]:
                player_relative = obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index]
                neutral_y, neutral_x = (player_relative == 3).nonzero()
                player_y, player_x = (player_relative == 1).nonzero()
                if not neutral_y.any() or not player_y.any():
                    action = actions.FUNCTIONS.no_op.id
                    params = []
                else:
                    player = [int(player_x.mean()), int(player_y.mean())]
                    closest, min_dist = None, None
                    for p in zip(neutral_x, neutral_y):
                        dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
                        if not min_dist or dist < min_dist:
                            closest, min_dist = p, dist
                    action = actions.FUNCTIONS.Move_screen.id
                    params = [[0], closest] # _SELECT_ALL = [0]
            else:
                action = actions.FUNCTIONS.select_army.id
                params = [[0]] # _SELECT_ALL = [0]

        self.states.append(np.array([obs.observation, action, params]))

        if len(self.states) == 64:
            new_file_name = str(uuid.uuid1())

            np.save("dataset_{}/{}".format(GAME, new_file_name), np.array(self.states))

            self.states = []

        return actions.FunctionCall(action, params)

__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import os
import numpy as np

from pysc2.lib import actions
from common.config import *


class Dataset:
    def __init__(self):
        self.input_observations = []
        self.input_available_actions = []
        self.output_actions = []
        self.output_params = []

    def load(self, path):
        print("Loading data...")
        for f in os.listdir(path):
            if f.find(".npy") != -1:
                file_name = f[:f.find(".npy")]
                states = np.load("{}/{}.npy".format(path, file_name))

                for results,action,param in states:
                    obs = [res for res in results] 
                    # rewards = [res.reward for res in results]
                    # dones = [res.last() for res in results]

                    self.input_observations.append(obs) # observations

                    output_size = len(actions.FUNCTIONS)
                    output_action = np.zeros(output_size)
                    output_action[action] = 1.0
                    self.output_actions.append(output_action) # one_hot action

                    self.output_params.append(param)

        assert len(self.input_observations) == len(self.output_actions) == len(self.output_params)

        self.input_observations = np.array(self.input_observations)
        # self.input_available_actions = np.array(self.input_available_actions)
        self.output_actions = np.array(self.output_actions)
        self.output_params = np.array(self.output_params)

        print("input observations: ", np.shape(self.input_observations))
        # print("input available actions ", np.shape(self.input_available_actions))
        print("output actions: ", np.shape(self.output_actions))
        print("output params: ", np.shape(self.output_params))

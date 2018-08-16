__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import os
import numpy as np
import pickle

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
            if f.find(".replay") != -1:
                file_name = f[:f.find(".replay")]
                #states = np.load("{}/{}.npy".format(path, file_name))
                states = pickle.load(open("{}/{}.replay".format(path, file_name), 'rb'))
                for results,action,param in states:
                    # print(results)
                    # obs = [res for res in results] 
                    # print(obs)
                    # rewards = [res.reward for res in results]
                    # dones = [res.last() for res in results]

                    # self.input_observations.append(results) # observations

                    # output_size = len(actions.FUNCTIONS)
                    # output_action = np.zeros(output_size)
                    # output_action[action] = 1.0 # one_hot action
                    self.input_observations.append(results)

                    self.output_actions.append([action]+param)

                    # self.output_params.append(param)

                # print(self.output_params)

        assert len(self.input_observations) == len(self.output_actions)# == len(self.output_params)

        # self.input_observations = np.array(self.input_observations) # dictionary
        # self.input_available_actions = np.array(self.input_available_actions)
        self.output_actions = np.array(self.output_actions)
        # self.output_params = np.array(self.output_params)

        print("input observations: ", np.shape(self.input_observations))
        # print("input available actions ", np.shape(self.input_available_actions))
        print("output actions: ", np.shape(self.output_actions))
        # print("output params: ", np.shape(self.output_params))

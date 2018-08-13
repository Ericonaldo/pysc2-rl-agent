__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import os
import numpy as np

from pysc2.lib import actions


class Dataset:
    def __init__(self):
        self.input_observations = []
        self.input_available_actions = []
        self.output_actions = []
        self.output_params = []

    def load(self, path):
        print "Loading data..."
        for f in os.listdir(path):
            if f.find(".npy") != -1:
                file_name = f[:f.find(".npy")]
                states = np.load("{}/{}.npy".format(path, file_name))

                for state in states:
                    self.input_observations.append(state[0]) # observations

                    output_size = len(actions.FUNCTIONS)
                    output_action = np.zeros(output_size)
                    output_action[state[1]] = 1.0
                    self.output_actions.append(output_action) # one_hot action

                    if np.shape(state[2]) == (2,): 
                        image_size = np.shape(state[0])[0]
                        point = [float(state[2][1][0]) / image_size, float(state[2][1][1]) / image_size] # state[2][1] 为target
                        self.output_params.append(point)
                    else:
                        self.output_params.append([0, 0])

        assert len(self.input_observations) == len(self.input_available_actions) == len(self.output_actions) == len(self.output_params)

        self.input_observations = np.array(self.input_observations)
        self.input_available_actions = np.array(self.input_available_actions)
        self.output_actions = np.array(self.output_actions)
        self.output_params = np.array(self.output_params)

        print "input observations: ", np.shape(self.input_observations)
        print "input available actions ", np.shape(self.input_available_actions)
        print "output actions: ", np.shape(self.output_actions)
        print "output params: ", np.shape(self.output_params)

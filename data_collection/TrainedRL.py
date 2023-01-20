from collections import namedtuple
from collections import OrderedDict
import time, os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.cmd_util import make_vec_env


class TrainedRL():
    def __init__(self, config):
        self.config = config

        self.state_dict = {'mean_resptime': 0, 'resptime': [0 for state_i in range(config['buffer_num'])]}
        self.overall_resptime_list = []

        self.action = 0

        self.rl_agent = PPO.load(config['model_path'])
    
    def update_state(self, state):
        self.overall_resptime_list.append(state[1])
        self.state_dict['resptime'] = self.state_dict['resptime'][1:]+[state[1]]
        self.state_dict['mean_resptime'] = np.mean(self.overall_resptime_list)
    
    def update_action(self):
        self.state_dict_env = OrderedDict([('mean_resptime', np.array(self.state_dict['mean_resptime']).reshape((1,1))), ('resptime', np.array(self.state_dict['resptime']).reshape((1,1,len(self.state_dict['resptime']))))])
        self.action, _states = self.rl_agent.predict(self.state_dict_env,deterministic=False)
        self.action = self.action[0]
        print('action: ',self.action,', from state: ',self.state_dict)
        return self.action



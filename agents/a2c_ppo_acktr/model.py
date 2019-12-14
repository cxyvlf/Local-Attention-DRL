import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 

from a2c_ppo_acktr.distribution import Categorical
from a2c_ppo_acktr.utils import init

class Flatten(nn.Module):
    def forward(self, x):
        # TO DO
        pass

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base in None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if 

    @property
    def is_recurrent(self):
        pass

    @property
    def recurrent_hidden_state_size(self):
        pass

    def forward(self, inputs, rnn_hxs, masks):
        pass

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        pass

    def get_value(self, inputs, rnn_hxs, masks):
        pass

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        pass

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        pass
    @property
    def is_recurrent(self):
        pass
    
    @property
    def recurrent_hidden_state_size(self):
        pass

    @property
    def output_size(self):
        pass

    def _forward_gru(self, x, hxs, masks):
        pass

class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        pass

    def forward(self, inputs, rnn_hxs, masks):
        pass


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        pass

    def forward(self, inputs, rnn_hxs, masks):
        pass
    

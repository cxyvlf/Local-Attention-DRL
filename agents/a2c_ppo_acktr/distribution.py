import torch
import torch.nn as nn 
import torch.nn.functional as F 

from a2c_ppo_acktr.utils import AddBias, init

# Categorical 
class FixedCategorical(torch.distributions.Categorical):
    # TO DO 
    def sample(self):
        pass

    def log_probs(self, actions):
        pass

    def mode(self):
        pass


class Categorical(nn.Module):
    # TO DO
    def __init__(self, num_inputs, num_outputs):
        pass

    def forward(self, x):
        pass



# # Normal
# class FixedNormal(torch.distributions.Normal):



# # Bernoulli
# class FixedBernoulli(torch.distributions.Bernoulli):
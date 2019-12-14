import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# TODO read

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        pass

    def update(self, rollouts):
        pass

    

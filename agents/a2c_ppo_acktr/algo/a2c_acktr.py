import torch
import torch.nn as nn
import torch.optim as optim

from a2c_ppo_acktr.algo.kfac import KFACOptimizer

# TODO

class A2C_ACKTR():
    def __init__(
                self,
                actor_critic,
                value_loss_coef,
                entropy_coef,
                lr=None,
                eps=None,
                alpha=None,
                max_grad_norm=None,
                acktr=False):
        
        pass

    def update(self, rollouts):
        pass



        

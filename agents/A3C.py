import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from ..schema_games.breakout import games


env_args = {
    'return_state_as_image': True,
}

env = games.StandardBreakout(**env_args)

state_space = env.observation_space.shape
n_action = env.action_space.n


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()



class AC_Net(nn.Module):
    def __init__(self, state_space, n_action):
        super(AC_Net, self).__init__()

    def forward(self, ):


    def choose_aciton(self):


    def loss_func(self):


class Worker(mp.Process):
    def __init__(self, global_AC, optimizer,  ,name):
        super(Worker, self).__init__()
        self.name = 
        self.env = games.StandardBreakout(**env_args)



    def run(self):


if __name__ == "__main__":
    global_AC = AC_Net(state_space, n_action)
    optimizer = SharedAdam(global_AC.parameters(), lr=0.0001) 
    workers = [Worker(global_AC, i) for i in range(mp.cpu_count)]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from A3C_utiles import v_wrap, set_init, push_and_pull, record

import sys
from os import path

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from schema_games.breakout import games

env_args = {
    'return_state_as_image': True,
}

n_channel = 3
n_action = 3

MAX_EP = 4000
UPDATE_GLOBAL_ITER =10
GAMMA = 0.9


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
    def __init__(self, n_channel, n_action):
        super(AC_Net, self).__init__()
        self.conv1 = nn.Conv2d(n_channel, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5, 128) # *see below
        self.critic_linear, self.actor_linear = nn.Linear(128, 1), nn.Linear(128, n_action)
        self.distribution = torch.distributions.Categorical
    
    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 5 * 5), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

    def choose_aciton(self, state):
        self.eval()
        _, logits = self.forward(state)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, state, action, value_t):
        self.train()
        values, logits = self.forward(state)
        
        TD_error = value_t - values
        c_loss = TD_error.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * TD_error.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, global_AC, optimizer, global_episode, global_episode_reward, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name

        self.global_episode, self.global_episode_reward, self.res_queue = global_episode, global_episode_reward, res_queue

        self.global_AC = global_AC
        self.optimizer = optimizer
        self.local_AC = AC_Net(n_channel, n_action)
        self.env = games.StandardBreakout(**env_args)

    def run(self):
        totel_step = 1
        while self.global_episode.value < MAX_EP:
            state = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            episode_reward = 0
            hx = torch.zeros(1, 256) if done else hx.detach()  # rnn activation vector

            while True:
                if self.name == 'w0':
                    self.env.render()
                
                action = self.local_AC.choose_aciton((v_wrap(state[None, :]), hx))

                state_, reward, done, _ = self.env.step(action)

                if done:
                    reward = -1    # TO DO 

                episode_reward += reward

                buffer_a.append(action)
                buffer_s.append(state)
                buffer_r.append(reward)

                if totel_step % UPDATE_GLOBAL_ITER == 0 or done:
                    push_and_pull(self.optimizer, self.local_AC, self.global_AC, done, state_, buffer_s, buffer_a, buffer_r, GAMMA)

                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.global_episode, self.global_episode_reward, episode_reward, self.res_queue, self.name)
                        break

                state = state_
                totel_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    global_AC = AC_Net(n_channel, n_action)
    global_AC.share_memory()

    optimizer = SharedAdam(global_AC.parameters(), lr=0.0001) 
    
    global_episode, global_episode_reward, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    
    workers = [Worker(global_AC, optimizer, global_episode, global_episode_reward, res_queue, i) for i in range(mp.cpu_count())]

    [w.start() for w in workers]

    [w.join() for w in workers]
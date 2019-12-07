import torch 
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def _flatten_helper(T, N, _tensor):
    pass

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size):
        pass

    def to(self, device):
        pass

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        pass

    def after_update(self):
        pass

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda, use_proper_time_limits=True):
        pass

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        pass

    def recurrent_generator(self, advantages, num_mini_batch):
        pass

    

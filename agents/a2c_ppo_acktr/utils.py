# TO DO
import glob
import os

import torch
import torch.nn as nn

class AddBias(nn.Module):
    # TO DO
    pass

def init(module, weight_init, bias_init, gain=1):
    # TO DO
    pass

def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
import numpy as np
import torch
import os
import random
import re
import time
import torchvision
import socket
import json

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed) -> None:
    """ set random seed for all related packages
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'backends'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LogWriter():
    def __init__(self, path, verbose=True, write_mode='a'):
        assert write_mode in ['a', 'w']
        self.verbose = verbose
        self.f = open(path, write_mode)

    def cprint(self, text):
        if self.verbose:
            print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def log_args(log_path, args):
    logfile = LogWriter(os.path.join(log_path, "log.txt"), verbose=True)
    # record time
    logfile.cprint("Execution time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    logfile.cprint("-"*15 + " args " + "-"*15) # separator
    for k, v in args.__dict__.items():
        logfile.cprint(f"{k} : {v}")
    logfile.cprint("-"*30) # separator
    logfile.cprint("Numpy: {}".format(np.__version__))
    logfile.cprint("Pytorch: {}".format(torch.__version__))
    logfile.cprint("torchvision: {}".format(torchvision.__version__))
    logfile.cprint("Cuda: {}".format(torch.version.cuda))
    logfile.cprint("hostname: {}".format(socket.gethostname()))
    logfile.cprint("="*50) # separator
    logfile.close()


def load_json(data_path):
    assert data_path.endswith('.json')
    with open(data_path,'r') as f:
        data = json.load(f)
    return data


def save_json(obj, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'w') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

def huber(x: torch.Tensor, delta: float = 1.0):
    return torch.where(torch.abs(x) < delta, 0.5 / delta * (x ** 2),  torch.abs(x) - 0.5 * delta)
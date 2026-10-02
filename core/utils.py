'''
* @name: utils.py
* @description: Other functions.
'''


import os
import random

import numpy as np
import torch


class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count):
        self.value = value
        self.value_sum += value * count
        self.count += count
        self.value_avg = self.value_sum / self.count


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(save_path, epoch, model, optimizer):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, 'epoch_{}.pth'.format(epoch))
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path)


def save_best_model(save_path,seed,model,optimizer,epoch):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path,'best_seed_{}.pth'.format(seed))
    states = {
        'epoch':epoch,
        'state_dict':model.state_dict(),
        'optimizer':optimizer.state_dict(),
    }
    torch.save(states,save_file_path)


def load_best_model(save_path,seed,model):
    save_file_path = os.path.join(save_path,'best_seed_{}.pth'.format(seed))
    if not os.path.exists(save_file_path):
        return None
    checkpoint = torch.load(save_file_path,map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint.get('epoch')

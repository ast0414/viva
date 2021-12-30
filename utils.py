import math
import os
import shutil

import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class CosineAnnealing(object):
    def __init__(self, start, stop, t_max, mode='up'):

        self.start = start
        self.t_max = t_max
        self.mode = mode

        self.scale = abs(stop - start)
        self.t_cur = -1
        self.value = start

    def step(self):
        self.t_cur += 1
        self.t_cur = min(self.t_cur, self.t_max)

        v = (1 - math.cos(self.t_cur / self.t_max * math.pi)) / 2
        if self.mode == 'up':
            v *= 1
        elif self.mode == 'down':
            v *= -1
        else:
            raise AttributeError("wrong mode")

        self.value = self.start + self.scale * v
        return self.value


def one_hot(x, label_size):
    out = torch.zeros(len(x), label_size).to(x.device)
    out[torch.arange(len(x)), x.squeeze()] = 1
    return out


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        directory = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(directory, 'best_checkpoint.pth'))

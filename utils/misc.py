import argparse
import random

import numpy as np
from pydoc import locate

import torch


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def import_model(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def import_class(name):
    return locate(name)


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def denormalize(label, class_idx, upper=100.0):
    label_ranges = {
        1: (21.6, 102.6),
        2: (12.3, 16.87),
        3: (8.0, 50.0),
        4: (8.0, 50.0),
        5: (46.2, 104.88),
        6: (49.8, 99.36)
    }
    label_range = label_ranges[class_idx]

    true_label = (label.float() / float(upper)) * (label_range[1] - label_range[0]) + label_range[0]
    return true_label


def normalize(label, class_idx, upper=100.0):
    label_ranges = {
        1: (21.6, 102.6),
        2: (12.3, 16.87),
        3: (8.0, 50.0),
        4: (8.0, 50.0),
        5: (46.2, 104.88),
        6: (49.8, 99.36)
    }
    label_range = label_ranges[class_idx]

    norm_label = ((label - label_range[0]) / (label_range[1] - label_range[0])) * float(upper)
    return norm_label

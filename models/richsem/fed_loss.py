import torch
import json
import numpy as np
from torch.nn import functional as F

def load_class_freq(
    path='datasets/lvis/lvis_v1_train_cat_info.json', 
    freq_weight=0.5):
    cat_info = json.load(open(path, 'r'))
    cat_info = torch.tensor(
        [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
    freq_weight = cat_info.float() ** freq_weight
    return freq_weight

def get_fed_loss_inds(gt_classes, num_sample_cats=50, C=1203, weight=None):
    appeared = torch.unique(gt_classes) # C'
    prob = appeared.new_ones(C).float()
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:] = weight.float().clone()
        prob[appeared] = 0
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared),
            replacement=False)
        appeared = torch.cat([appeared, more_appeared])
    return appeared
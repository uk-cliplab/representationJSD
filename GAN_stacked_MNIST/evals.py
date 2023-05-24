### Code from https://github.com/ritheshkumar95/energy_based_generative_models/blob/master/scripts/evals.py


import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from MNISTclassifier import Net
import os
from pathlib import Path
from utils import get_truncated_noise
import torchvision.transforms as T
import torch.nn.functional as F

def KLD(p, q):
    if 0 in q:
        raise ValueError
    return sum(_p * np.log(_p/_q) for (_p, _q) in zip(p, q) if _p != 0)


class ModeCollapseEval(object):
    def __init__(self, n_stack, z_dim):
        self.classifier = Net().cuda()
        self.classifier.load_state_dict(torch.load('saved_models/pretrained_classifier.pt'))
        self.n_stack = n_stack
        self.n_samples = 26 * 10 ** n_stack
        self.z_dim = z_dim

    def count_modes(self, netG):
        counts = np.zeros([10] * self.n_stack)
        n_batches = max(1, self.n_samples // 1000)
        for i in tqdm(range(n_batches)):
            with torch.no_grad():
                z = get_truncated_noise(1000, self.z_dim, 0.5).cuda()
                # z = torch.randn(1000, self.z_dim).cuda()
                x_fake = netG(z) * .5 + .5

                # reshape tensors to 28 by 28
                x_fake  = F.interpolate(x_fake, size=28) # resize to 28 by 28 (images were resized to 64x64 to use DCGAN)

                x_fake = x_fake.view(-1, 1, 28, 28)
                classes = F.softmax(self.classifier(x_fake), -1).max(1)[1]
                classes = classes.view(1000, self.n_stack).cpu().numpy()

                for line in classes:
                    counts[tuple(line)] += 1

        n_modes = 10 ** self.n_stack
        true_data = np.ones(n_modes) / float(n_modes)
        num_modes_cap = len(np.where(counts > 0)[0])
        counts = counts.flatten() / counts.sum()
        kld = KLD(counts, true_data)
        print("No. of modes captured: ", num_modes_cap)
        print('Reverse KL: ', kld)
        return num_modes_cap, kld
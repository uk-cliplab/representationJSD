import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision

import random
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt

from scipy.stats import truncnorm
def get_truncated_noise(n_samples, z_dim, truncation):
    '''
    Function for creating truncated noise vectors: Given the dimensions (n_samples, z_dim)
    and truncation value, creates a tensor of that shape filled with random
    numbers from the truncated normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        truncation: the truncation value, a non-negative scalar
    '''
    #### START CODE HERE ####
    truncated_noise = truncnorm.rvs(-truncation,truncation, size=(n_samples, z_dim))
    #### END CODE HERE ####
    noise = torch.Tensor(truncated_noise)
    # reshape to add two channels
    noise = noise.view(n_samples, z_dim, 1, 1)
    return noise

class stacked_mnist_multiview_dataset(Dataset):
    """
    Implementation of a multiview dataset. 
    Inputs:
        raw_data - raw data to create from. optional if dataset is saved
        dataset_id - name of folder to which the multiview dataset will be stored/loaded from
        
    Outputs:
        Multiview dataset that can be given to a Dataloader.
    """

    def __init__(self, raw_data=None, check_for_saved=True, is_testset=False):
        # setup config variables

        self.raw_data = raw_data
        self.is_testset = is_testset
        self.custom_labels = None
        dataset_id = "stacked_mnist"

        # setup folder locations
        if self.is_testset:
            self.label_path = "data/%s/labels_test.npy" % dataset_id
            self.data_path = "data/%s/paired_imgs_test.npy" % dataset_id
            self.per_type_size = 10
        else:
            self.label_path = "data/%s/labels.npy" % dataset_id
            self.data_path = "data/%s/paired_imgs.npy" %dataset_id
            self.per_type_size = 60

        # check if data exists and load if allowed
        if check_for_saved and os.path.exists(self.label_path) and os.path.exists(self.data_path):
            self.data, self.labels = self.load_dataset()
        
        # create dataset from scratch
        else:
            if not os.path.exists("data/%s" % dataset_id):
                os.mkdir("data/%s" % dataset_id)
    
            if self.raw_data is None:
                self.raw_data = self.get_raw_mnist()


            self.stack_pairings, self.stack_labels = self.constructStacks()

            self.data, self.labels = self.constructData()
            self.save_dataset()
            

    def constructStacks(self):
        stack_pairings, stack_labels = [], []

        if (torch.is_tensor(self.raw_data.targets)):
            targets = np.array([i.item() for i in self.raw_data.targets])
        else:
            targets = np.array([i for i in self.raw_data.targets])

        lbl_set = set(list(targets))

        stack_types = itertools.product(lbl_set, lbl_set, lbl_set)
        for tidx, t in enumerate(stack_types):
            digit_a, digit_b, digit_c = t

            digit_a = np.random.choice(np.where(targets == digit_a)[0], size=self.per_type_size)
            digit_b = np.random.choice(np.where(targets == digit_b)[0], size=self.per_type_size)
            digit_c = np.random.choice(np.where(targets == digit_c)[0], size=self.per_type_size)
 
            for idx in range(self.per_type_size):
                pairing = (digit_a[idx], digit_b[idx], digit_c[idx])
                stack_pairings.append(pairing)
                stack_labels.append(tidx)

        return stack_pairings, stack_labels

    def constructData(self):
        """
        Applies transforms to data
        Outputs:
            List of data tuples s.t. each tuple is (rotated_stack, normal_stack)
            1-D list of labels
        """
        num_data = len(self.stack_labels)
        data_size = self.raw_data[0][0].shape
        
        data = np.zeros((num_data, 3, *data_size[1:]),dtype=np.float32)
        labels = np.zeros(num_data, dtype=int)

        for idx in range(num_data):
            normal_stack_raw_indices = np.array(self.stack_pairings[idx])

            normal_instance = torch.cat((self.raw_data[normal_stack_raw_indices[0]][0],
                    self.raw_data[normal_stack_raw_indices[1]][0],
                    self.raw_data[normal_stack_raw_indices[2]][0]))


            
            data[idx, :] = normal_instance
            labels[idx] = self.stack_labels[idx]
        
        return data, labels
    

        
    def __safe_print__(self, message):
        """
        Prints a message if class has show_progress set
        """
        if (self.show_progress):
            print(message)
        
    def __len__(self):
        """
        Overload of len() function in superclass
        How many data points are in this dataset
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Overload of getitem() function in superclass
        Gets data at position idx. Of the form (rotated_instance, noisy_instance), label
        """
        label = self.labels[idx]
        items = torch.from_numpy(self.data[idx, :])
        return items, label
    
    def save_dataset(self):
        """
        Saves dataset to file.
        """
        
        np.save(self.data_path, self.data)
        np.save(self.label_path, self.labels)
        
    def load_dataset(self):
        """
        Load dataset from file
        """
        labels = np.load(self.label_path).astype(np.int)
        data = np.load(self.data_path).astype(np.float32)
        return data, labels

    def get_raw_mnist(self):
        """
        For rot_noisy MNIST, no need to supply raw_data to __init__. We can load it here
        """
        if not self.is_testset:
            return datasets.MNIST(root='data', 
                                   train=True, 
                                   transform=transforms.Compose(
                                       [transforms.Resize(64), transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)
                                     ]),
                                   download=True)
        # resize to use DCGAN architecture
        else:
            return datasets.MNIST(root='data', 
                                        train=False, 
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(),
                                            ]),
                                        download=True)

    def get_custom_labels(self):
        return self.custom_labels if self.custom_labels is not None else self.labels

def visualize_dataset(data_loader, img_size=(1, 28, 28), sidesize=8):
    """
    Plot small batch of multiview dataset. 25 image grid, Side-by-side of view 1 and view 2
    """

    imgs = torch.zeros((sidesize**2, *img_size))

    
    batch_data, batch_labels =next(iter(data_loader))
    print(batch_data.shape)
    

    for i in range(sidesize**2):
        imgs[i] = batch_data[i].reshape(img_size)

    imgs = normalize_images(imgs, 0.5, 0.5)


    fig, ax = plt.subplots(1,1, figsize=[4,4])

    img_grid = torchvision.utils.make_grid(imgs, nrow=sidesize)
    ax.imshow(img_grid.permute((1, 2, 0)))

    ax.tick_params(bottom = False)
    ax.tick_params(left = False)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    plt.show()


def normalize_images(imgs, mean, std):
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/std, 1/std, 1/std ]),
                                    transforms.Normalize(mean = [ -1*mean, -1*mean, -1*mean ],
                                                        std = [ 1., 1., 1. ]),
                                ])

    inv_tensor = invTrans(imgs)
    return inv_tensor
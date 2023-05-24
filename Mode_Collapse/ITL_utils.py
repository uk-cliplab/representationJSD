
import torch
import numpy as np


def vonNeumannEntropy(K, lowRank = False, rank = None):
    n = K.shape[0]
    ek, _ = torch.linalg.eigh(K)
    if lowRank:
        ek_lr = torch.zeros_like(ek)
        ek_lr[-rank:] = ek[-rank:]
        remainder = ek.sum() - ek_lr.sum()
        ek_lr[:(n-rank)] = remainder/(n-rank)
        mk = torch.gt(ek_lr, 0.0)
        mek = ek_lr[mk]
    else:
        mk = torch.gt(ek, 0.0)
        mek = ek[mk]

    mek = mek/mek.sum()   
    H = -1*torch.sum(mek*torch.log2(mek))
    return H


def deep_JSD(X,Y,model):
    phiX = model(X)
    phiY = model(Y)
    # Creating the mixture of both distributions
    # phiZ =  torch.cat((phiX,phiY))
    covX = torch.matmul(torch.t(phiX),phiX)
    covY = torch.matmul(torch.t(phiY),phiY)
    Hx = vonNeumannEntropy(covX)
    Hy = vonNeumannEntropy(covY)
    Hz = vonNeumannEntropy((covX+covY)/2)
    JSD =  (Hz - 0.5*(Hx + Hy))
    return JSD

def JSD_cov(covX,covY):
    Hx = vonNeumannEntropy(covX)
    Hy = vonNeumannEntropy(covY)
    Hz = vonNeumannEntropy((covX+covY)/2)
    JSD =  (Hz - 0.5*(Hx + Hy))
    return JSD

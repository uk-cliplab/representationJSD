import numpy as np
import torch
from sklearn.utils import check_random_state
import pickle

def sample_blobs(samples_per_blob, rows=3, cols=3, rs=None):
    """Generate Blob-D for testing type-II error (or test power)."""
    # Covariances for the modes of Q
    N1 = 9*samples_per_blob
    sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
    sigma_mx_2 = np.zeros([9,2,2])
    for i in range(9):
        sigma_mx_2[i] = sigma_mx_2_standard
        if i < 4:
            sigma_mx_2[i][0 ,1] = -0.02 - 0.002 * i
            sigma_mx_2[i][1, 0] = -0.02 - 0.002 * i
        if i==4:
            sigma_mx_2[i][0, 1] = 0.00
            sigma_mx_2[i][1, 0] = 0.00
        if i>4:
            sigma_mx_2[i][1, 0] = 0.02 + 0.002 * (i-5)
            sigma_mx_2[i][0, 1] = 0.02 + 0.002 * (i-5)

    rs = check_random_state(rs)
    mu = np.zeros(2)
    sigma = np.eye(2) * 0.03
    X = rs.multivariate_normal(mu, sigma, size=N1)
    Y = rs.multivariate_normal(mu, np.eye(2), size=N1)
    # assign to blobs
    X[:, 0] += rs.randint(rows, size=N1)
    X[:, 1] += rs.randint(cols, size=N1)
    Y_row = rs.randint(rows, size=N1)
    Y_col = rs.randint(cols, size=N1)
    locs = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    for i in range(9):
        corr_sigma = sigma_mx_2[i]
        L = np.linalg.cholesky(corr_sigma)
        ind = np.expand_dims((Y_row == locs[i][0]) & (Y_col == locs[i][1]), 1)
        ind2 = np.concatenate((ind, ind), 1)
        Y = np.where(ind2, np.matmul(Y,L) + locs[i], Y)
    X = torch.tensor(X,dtype = torch.float)
    Y = torch.tensor(Y,dtype = torch.float)
    return X, Y

def HDGM(samples_per_cluster, dimension, seed):
    n = samples_per_cluster
    d = dimension
    rho = 0.5
    Num_clusters = 2 # number of modes
    mu_mx = np.zeros([Num_clusters,d])
    mu_mx[1] = mu_mx[1] + 0.5
    sigma_mx_1 = np.identity(d)
    sigma_mx_2 = [np.identity(d),np.identity(d)]
    sigma_mx_2[0][0,1] = rho
    sigma_mx_2[0][1,0] = rho
    sigma_mx_2[1][0,1] = -rho
    sigma_mx_2[1][1,0] = -rho
    s1 = np.zeros([n*Num_clusters, d])
    s2 = np.zeros([n*Num_clusters, d])
        # Generate HDGM-D
    for i in range(Num_clusters):
        np.random.seed(seed=seed + i + n)
        s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
    for i in range(Num_clusters):
        np.random.seed(seed=seed + 1 + i + n)
        s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n)
    X = torch.tensor(s1,dtype = torch.float)
    Y = torch.tensor(s2,dtype = torch.float)
    return X, Y

def Higgs(samples, seed):
    np.random.seed(seed)
    n = samples
    # Load data
    data = pickle.load(open('../data/HIGGS_TST.pckl', 'rb'))
    dataX = data[0]
    dataY = data[1]
    # Generate Higgs (P,Q)
    N1_T = dataX.shape[0]
    N2_T = dataY.shape[0]
    ind1 = np.random.choice(N1_T, n, replace=False)
    ind2 = np.random.choice(N2_T, n, replace=False)
    s1 = dataX[ind1,:4]
    s2 = dataY[ind2,:4]
    X = torch.tensor(s1,dtype = torch.float)
    Y = torch.tensor(s2,dtype = torch.float)
    return X, Y



def gaussian_kernel(x,y,sigma):
    dists = (x-y)**2
    dists_norm = dists.sum(axis = 1)
    # print(dists_norm)
    return np.exp(-(1/(2*sigma**2))*(dists_norm))

def gaussian_kernel_kde(samples, data,sigma):
    N = data.shape[0]
    kde = 0
    for i in range(N):
        kde +=  gaussian_kernel(samples,data[i:i+1,:],sigma) 
    kde = kde
    return kde


def deep_kernel(x,y,model):
    x = torch.tensor(x,dtype = torch.float)
    y = torch.tensor(y,dtype = torch.float)
    
    phiX = model(x)
    phiY = model(y)

    Kxy = torch.matmul(phiX,torch.t(phiY)) 
    # Kxy = torch.matmul(phiX,torch.t(phiY))
    kde = Kxy
    return kde
    
def deep_kernel_kde(samples, data, model):
    N = data.shape[0]
    kde = 0
    for i in range(N):
        kde +=  deep_kernel(samples,data[i:i+1,:],model) 
    kde = 1/N*kde
    return kde


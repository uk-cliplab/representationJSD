
import torch
import numpy as np

from torch.autograd import grad
from models import RFF_layer

# imports from representation-itl library
import repitl.kernel_utils as ku
import repitl.matrix_itl as itl
import repitl.divergences as div
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

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

def nonInformative(K):
    n = K.shape[0]
    avg = (K.sum()-n) / (n**2-n)
    diag  = torch.diag(torch.ones(n) - avg)
    off_diag = avg*torch.ones([n,n])
    N = diag + off_diag
    return N

def nonInformativeAlphaEntropy(N, alpha=1.01):
    """
    Computes the entropy for a non-informative matrix:
    
    Args: (N x N) non-informative matrix. All the off-diagonal
          elements must be equal and between 0 and 1. 
    """
    n  = N.shape[0]
    Evni = torch.zeros(n)
    avg = N[0,1]
    Evni[0] = 1/n + (n-1)*avg/n
    Evni[1:] = (1-Evni[0])/(n-1)
    Hni = (1/(1-alpha))*torch.log(torch.sum(Evni**alpha))
    return Hni

def JSD_bound(X,Y,w1 =0.5, w2 = 0.5, sigma = 0.5):
    N = X.shape[0]
    M = Y.shape[0]
    # Creating indicator or label variable
    l = torch.ones(N+M, dtype=torch.long, device=X.device)
    l[N:] = 0
    # One-hot encoding the label variable
    L = torch.nn.functional.one_hot(l).type(X.dtype)
    # Creating the mixture of the two distributions
    Z = torch.cat((X,Y))
    # Kernels
    Kx = ku.gaussianKernel(X,X, sigma)
    Ky = ku.gaussianKernel(Y,Y, sigma)
    Kz = ku.gaussianKernel(Z,Z, sigma)
    # Kernel of the labels  
    Kl = torch.matmul(L, L.t())
    # Computing divergence
    Hx = vonNeumannEntropy(Kx)
    Hy = vonNeumannEntropy(Ky)
    #
    Kz_nonInformative = nonInformative(Kz)
    Hj_nonInformative = vonNeumannEntropy(Kz_nonInformative*Kl)
    JSD_info =  (Hj_nonInformative - 0.5*(Hx + Hy)) - np.log(2)
    return JSD_info

def JSD_difference(covX, covY,covXperm,covYperm,w1 =0.5, w2 = 0.5):

    # Computing divergence
    Hx = vonNeumannEntropy(covX)
    Hy = vonNeumannEntropy(covY)
    Hxperm = vonNeumannEntropy(covXperm)
    Hyperm = vonNeumannEntropy(covYperm)
    # FIGURE OUT WHO IS THE NON-INFORMATIVE COVARIANCE
    JSD_info =  0.5*(Hxperm + Hyperm)- 0.5*(Hx + Hy)
    return JSD_info

def JSD_difference_(X, Y,model):
    phiX = model(X)
    phiY = model(Y)
    phiZ =  torch.cat((phiX,phiY))

    # permuting the samples
    idxPerm = torch.randperm(len(phiZ))
    phiZperm = phiZ[idxPerm,:]
    phiXperm = phiZperm[:X.shape[0],:]
    phiYperm = phiZperm[-Y.shape[0]:,:]
    
    # We'll construct the covariance operators
    covX = torch.matmul(torch.t(phiX),phiX)
    covY = torch.matmul(torch.t(phiY),phiY)
    covXperm = torch.matmul(torch.t(phiXperm),phiXperm)
    covYperm = torch.matmul(torch.t(phiYperm),phiYperm)

    # Computing divergence
    Hx = vonNeumannEntropy(covX)
    Hy = vonNeumannEntropy(covY)
    Hxperm = vonNeumannEntropy(covXperm)
    Hyperm = vonNeumannEntropy(covYperm)
    JSD_info =  0.5*(Hxperm + Hyperm)- 0.5*(Hx + Hy)
    return JSD_info

def QJSD(X,Y, sigma = 0.5, w1 =0.5, w2 = 0.5,reduced_rank = False,top = None):
    N = X.shape[0]
    M = Y.shape[0]
    # Creating the mixture of the two distributions
    Z = torch.cat((X,Y))
    # Kernels
    Kx = ku.gaussianKernel(X,X, sigma)
    Ky = ku.gaussianKernel(Y,Y, sigma)
    Kz = ku.gaussianKernel(Z,Z, sigma)
    # Kernel of the labels  
    # Computing divergence

    if reduced_rank:
        Hz = vonNeumannEntropy(Kz, reduced_rank=True, rank= N, top = top) # this can be either X.shape or Y.shape
        Hx = vonNeumannEntropy(Kx,reduced_rank=True, rank = N, top = top)
        Hy = vonNeumannEntropy(Ky,reduced_rank=True, rank =N, top = top)
    else:
        Hz = vonNeumannEntropy(Kz)
        Hx = vonNeumannEntropy(Kx)
        Hy = vonNeumannEntropy(Ky)
    #

    JSD =  (Hz - 0.5*(Hx + Hy))
    return JSD

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

def JSD_difference_augmented(X,Y,model , n_perm = 20):
    phiX = model(X)
    phiY = model(Y)
    # Creating the mixture of both distributions
    phiZ =  torch.cat((phiX,phiY))



    d = phiX.shape[1]

    covXperm = torch.zeros(d,d,n_perm)
    covYperm = torch.zeros(d,d,n_perm)
    for i in range(n_perm):
        # Creating the mixture of both distributions
        phiZ =  torch.cat((phiX,phiY))


        # permuting the samples
        idxPerm = torch.randperm(len(phiZ))
        phiZperm = phiZ[idxPerm,:]
        phiXperm = phiZperm[:X.shape[0],:]
        phiYperm = phiZperm[-Y.shape[0]:,:]
        covXperm[:,:,i] = torch.matmul(torch.t(phiXperm),phiXperm)
        covYperm[:,:,i] = torch.matmul(torch.t(phiYperm),phiYperm)
    covXperm_avg = covXperm.mean(dim = 2)
    covYperm_avg = covYperm.mean(dim = 2)
    covX = torch.matmul(torch.t(phiX),phiX)
    covY = torch.matmul(torch.t(phiY),phiY)
    Hx = vonNeumannEntropy(covX)
    Hy = vonNeumannEntropy(covY)
    Hzx = vonNeumannEntropy(covXperm_avg)
    Hzy = vonNeumannEntropy(covYperm_avg)
    JSD =  (0.5*(Hzx + Hzy) - 0.5*(Hx + Hy))
    return JSD  
    

def JRD_bound(X,Y, sigma = 0.5,alpha = 0.1):
    N = X.shape[0]
    M = Y.shape[0]
    # Creating indicator or label variable
    l = torch.ones(N+M, dtype=torch.long, device=X.device)
    l[N:] = 0
    # One-hot encoding the label variable
    L = torch.nn.functional.one_hot(l).type(X.dtype)
    # Creating the mixture of the two distributions
    Z = torch.cat((X,Y))
    # Kernels
    Kx = ku.gaussianKernel(X,X, sigma)
    Ky = ku.gaussianKernel(Y,Y, sigma)
    Kz = ku.gaussianKernel(Z,Z, sigma)
    # Kernel of the labels  
    Kl = torch.matmul(L, L.t())
    # Computing divergence
    Hx = itl.matrixAlphaEntropy(Kx,alpha = alpha)
    Hy = itl.matrixAlphaEntropy(Ky,alpha = alpha)
    Kz_nonInformative = nonInformative(Kz)
    Hj_nonInformative = itl.matrixAlphaEntropy(Kz_nonInformative*Kl, alpha = alpha)
    JRD_info =  (Hj_nonInformative - 0.5*(Hx + Hy)) - np.log(2)
    return JRD_info

def QJRD(X,Y, sigma = 0.5,alpha = 0.1):
    N = X.shape[0]
    M = Y.shape[0]
    # Creating indicator or label variable
    l = torch.ones(N+M, dtype=torch.long, device=X.device)
    l[N:] = 0
    # One-hot encoding the label variable
    L = torch.nn.functional.one_hot(l).type(X.dtype)
    # Creating the mixture of the two distributions
    Z = torch.cat((X,Y))
    # Kernels
    Kx = ku.gaussianKernel(X,X, sigma)
    Ky = ku.gaussianKernel(Y,Y, sigma)
    Kz = ku.gaussianKernel(Z,Z, sigma)
    # Kernel of the labels  
    Kl = torch.matmul(L, L.t())
    # Computing divergence
    Hx = itl.matrixAlphaEntropy(Kx,alpha = alpha)
    Hy = itl.matrixAlphaEntropy(Ky,alpha = alpha)
    Hz = itl.matrixAlphaEntropy(Kz,alpha = alpha)
    
    JRD =  (Hz - 0.5*(Hx + Hy)) 
    return JRD

def mmd(x, y, sigma):
    # compare kernel MMD paper and code:
    # A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012)
    # http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
    # x shape [n, d] y shape [m, d]
    # n_perm number of bootstrap permutations to get p-value, pass none to not get p-value
    n, d = x.shape
    m, d2 = y.shape
    assert d == d2
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    # note that sigma should be squared in the RBF to match the Gretton et al heuristic
    k = torch.exp((-1 / (2 * sigma ** 2)) * dists ** 2)
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    # mmd unbiased does not sum the diagonal terms.
    mmd = (k_x.sum() - n) / (n * (n - 1)) + (k_y.sum() - m) / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    return mmd



def bestSigmaDivergence(X,Y,typeDivergence = 'JSD', alpha = 0.99, learning_rate = 0.001, is_cuda = True,n_epochs = 50, initial_sigma = 'distance', validation = False, Xval = None, Yval = None):
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    # estimating an initial guess of sigma
    XY = torch.cat((X,Y))
    numSamples = X.shape[0] # + Y.shape[0]
    d_in = X.shape[1]
    # use this sigma_init for HDGM based on the dimension
    if initial_sigma == 'dimension':
        sigma_init = 2*torch.tensor(np.sqrt(2 * d_in))
    elif initial_sigma == 'distance':
        # Use this sigma_init for the blobs dataset
        dists = ku.squaredEuclideanDistance(XY, XY)
        sigma_init = 0.2*torch.sqrt(torch.sum(dists) / ( ((numSamples*2)**2 - (numSamples*2)) * 2 ))
    
    sigma = sigma_init.clone().detach()
    # Locating the model and data in the corresponding device
    if is_cuda:
        device = torch.device('cuda:0')    
    else:
        device = torch.device("cpu")

    X = X.to(device)
    Y = Y.to(device)
    if validation:
        Xval = Xval.to(device)
        Yval = Yval.to(device)

    sigma = sigma.to(device).requires_grad_()

    # Defining optimizer

    optimizer = torch.optim.Adam([sigma] , lr = learning_rate)
    scheduler = StepLR(optimizer, 
                   step_size = 200, # Period of learning rate decay
                   gamma = 0.5) # Multiplicative factor of learning rate decay


    log_dir = "runs/sigma/HDGM_sigma_"+ str(sigma_init.detach().item()) + str(numSamples) + "lr_"+str(learning_rate)#+  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)
    for epoch in range(n_epochs):
        # Creating the mixture of both distributions
        Z =  torch.cat((X,Y))

        # permuting the samples
        idxPerm = torch.randperm(len(Z))
        Zperm = Z[idxPerm,:]
        Xperm = Zperm[:X.shape[0],:]
        Yperm = Zperm[-Y.shape[0]:,:]
        
        # We'll construct the kernels or covariance operators
        Kx = ku.gaussianKernel(X,X, sigma)
        Ky = ku.gaussianKernel(Y,Y, sigma)
        Kxperm = ku.gaussianKernel(Xperm,Xperm, sigma)
        Kyperm = ku.gaussianKernel(Yperm,Yperm, sigma)
        
        # estimating the difference between the JSD of the real distributions and the permutations
        if typeDivergence == 'JSD':
            loss = -1*JSD_difference(Kx, Ky,Kxperm,Kyperm)
        # if typeDivergence == 'JRD'
        #   loss = -1*JRD_difference(Kx, Ky,Kxperm,Kyperm,alpha = alpha)
        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()
        scheduler.step()
        if validation:
            loss_val = validate_sigma(Xval, Yval, sigma.detach())
            writer.add_scalar('sigma', sigma.detach(), epoch)
            writer.add_scalars('J(w)', {'training':-1*loss, 'validation': -1*loss_val}, epoch)
    
    return sigma.cpu().detach()
    
def bestSigmaDivergenceRFF(X,Y,typeDivergence = 'JSD', 
    n_RFF = 100, 
    freezeRFF = False, 
    alpha = 0.99, 
    learning_rate = 0.001, 
    is_cuda = True,
    n_epochs = 50, 
    initial_sigma = 'distance', 
    validation = False,
    Xval = None, 
    Yval = None):

    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    # estimating an initial guess of sigma
    XY = torch.cat((X,Y))
    numSamples = X.shape[0] # + Y.shape[0]
    d_in = X.shape[1]
    

    # use this sigma_init for HDGM based on the dimension
    if initial_sigma == 'dimension':
        sigma_init = torch.tensor(np.sqrt(2 * d_in))
    elif initial_sigma == 'distance':
        # Use this sigma_init for the blobs dataset
        dists = ku.squaredEuclideanDistance(XY, XY)
        sigma_init = 0.2*torch.sqrt(torch.sum(dists) / ( ((numSamples*2)**2 - (numSamples*2)) * 2 ))
        
    sigma = sigma_init.clone().detach()
    
    # Locating the model and data in the corresponding device
    if is_cuda:
        device = torch.device('cuda:0')
        RFFestimator = RFF_layer(d_in, n_RFF,sigma_init,variant = 'cosine_sine',freezeRFF=freezeRFF).to(device) # just learn the bandwidth       
    else:
        device = torch.device("cpu")
        RFFestimator = RFF_layer(d_in, n_RFF,sigma_init,variant = 'cosine_sine',freezeRFF=freezeRFF).to(device)

    X = X.to(device)
    Y = Y.to(device)

    if validation:
        Xval = Xval.to(device)
        Yval = Yval.to(device)

    

    # Defining optimizer
    
    optimizer  = torch.optim.Adam(list(RFFestimator.parameters()) , lr = learning_rate)
    # log_dir = "runs/fit/RFF_" + str(numSamples) + "lr_"+str(learning_rate)#+  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # writer = SummaryWriter(log_dir)
    
    for epoch in range(n_epochs):
        loss = -1*deep_JSD(X,Y,RFFestimator)
        # # Creating the mixture of both distributions
        # phiX = RFFestimator(X)
        # phiY = RFFestimator(Y)
        # phiZ =  torch.cat((phiX,phiY))

        # # permuting the samples
        # idxPerm = torch.randperm(len(phiZ))
        # phiZperm = phiZ[idxPerm,:]
        # phiXperm = phiZperm[:X.shape[0],:]
        # phiYperm = phiZperm[-Y.shape[0]:,:]
       
        # # We'll construct the covariance operators
        # covX = torch.matmul(torch.t(phiX),phiX)
        # covY = torch.matmul(torch.t(phiY),phiY)
        # covXperm = torch.matmul(torch.t(phiXperm),phiXperm)
        # covYperm = torch.matmul(torch.t(phiYperm),phiYperm)

        
        # # estimating the difference between the JSD of the real distributions and the permutations
        # if typeDivergence == 'JSD':
        #     loss = -1*JSD_difference(covX, covY,covXperm,covYperm)
        # # if typeDivergence == 'JRD'
        # #   loss = -1*JRD_difference(Kx, Ky,Kxperm,Kyperm,alpha = alpha)
        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()
        if validation:
            loss_val = validate_model(Xval, Yval, RFFestimator)
            # writer.add_scalar('J(w)', -1*loss_val, epoch)
            # writer.add_scalars('J(w)', {'training':-1*loss, 'validation': -1*loss_val}, epoch)
            
    
    return RFFestimator

def validate_model(Xval, Yval, model):
    phiX = model(Xval)
    phiY = model(Yval)
    

    # Creating the mixture of both distributions
    phiZ =  torch.cat((phiX,phiY))

    # permuting the samples
    idxPerm = torch.randperm(len(phiZ))
    phiZperm = phiZ[idxPerm,:]
    phiXperm = phiZperm[:Xval.shape[0],:]
    phiYperm = phiZperm[-Yval.shape[0]:,:]
    
    # We'll construct the covariance operators


    covX = torch.matmul(torch.t(phiX),phiX)
    covY = torch.matmul(torch.t(phiY),phiY)
    covXperm = torch.matmul(torch.t(phiXperm),phiXperm)
    covYperm = torch.matmul(torch.t(phiYperm),phiYperm)
    
    # estimating the difference between the JSD of the real distributions and the permutations
    loss = -1*JSD_difference(covX, covY,covXperm,covYperm)
    return loss
def validate_sigma(Xval, Yval, sigma):
    Z =  torch.cat((Xval,Yval))

    # permuting the samples
    idxPerm = torch.randperm(len(Z))
    Zperm = Z[idxPerm,:]
    Xperm = Zperm[:Xval.shape[0],:]
    Yperm = Zperm[-Yval.shape[0]:,:]
    
    # We'll construct the kernels or covariance operators
    Kx = ku.gaussianKernel(Xval,Xval, sigma)
    Ky = ku.gaussianKernel(Yval,Yval, sigma)
    Kxperm = ku.gaussianKernel(Xperm,Xperm, sigma)
    Kyperm = ku.gaussianKernel(Yperm,Yperm, sigma)
    # estimating the difference between the JSD of the real distributions and the permutations
    loss = -1*JSD_difference(Kx, Ky,Kxperm,Kyperm)
    return loss

def optimizedDivergence(X,Y,typeDivergence = 'JSD',alpha = 0.99, epsilon = 0.01,max_iter = 10,print_results = False):
    
    XY = torch.cat((X,Y))
    numSamples = X.shape[0] 
    dists = ku.squaredEuclideanDistance(XY, XY)
    sigma_ = torch.sqrt(torch.sum(dists) / ( ((numSamples*2)**2 - (numSamples*2)) * 2 ))

    # We'll try Four different initial guests [0.25, 0.5, 1.0, 2.0]*sigma_init 
    # We'll also store the values with no optimization in the grid search

    factor = np.array([0.1,0.25,0.5,1.0])
    sigmas = sigma_.item()*factor
    divergence = np.zeros_like(sigmas)
    sigma_final = np.zeros_like(sigmas)

    for i,c in enumerate(factor):
        sigma_init = c*sigma_

        sigma1 = sigma_init.clone().detach().requires_grad_(True)
        sigma2 = (sigma_init + 0.1).clone().detach().requires_grad_(True) 


        for j in range(max_iter):
            if torch.abs(sigma1 - sigma2)< epsilon:
                break

            if typeDivergence == 'JSD':
                div1 = JSD_bound(X,Y,sigma = sigma1)
                div2 = JSD_bound(X,Y,sigma = sigma2)
            elif typeDivergence == 'JRD':
                div1 = JRD_bound(X,Y,sigma = sigma1, alpha = alpha)
                div2 = JRD_bound(X,Y,sigma = sigma2, alpha = alpha)
            elif typeDivergence == 'MMD':
                div1 = mmd(X,Y,sigma = sigma1)
                div2 = mmd(X,Y,sigma = sigma2)
            else:
                print('Got an invalid type of divergence')
                break

            df_sigma1, = grad(outputs=div1, inputs=sigma1)
            df_sigma2, = grad(outputs=div2, inputs=sigma2)
            df2 = (df_sigma2 - df_sigma1)/(sigma2-sigma1)
            if (df2>0):
                sigma = 0.5*sigma1.clone() # if the initial guest is too big, that the cost function is in a convex area, make it smaller (half)        
                sigma1 = sigma.clone()
                sigma2 = sigma.clone() + 0.1
            else:
                sigma = sigma2 - df_sigma2/df2 # secant method
                sigma1 = sigma2.clone()
                sigma2 = sigma.clone()
            if print_results == True:
                print(sigma.detach().item(),div2.detach().item())
        divergence[i] = div2.detach().item()
        sigma_final[i] = sigma.detach().item()
    
    best_sigma = sigma_final[np.argmax(divergence)]

    
    if typeDivergence == 'JSD':
        max_divergence =  QJSD(X,Y,sigma = best_sigma)
    elif typeDivergence == 'JRD':
        max_divergence =  QJRD(X,Y,sigma = best_sigma,alpha = alpha)
    elif typeDivergence == 'MMD':
        max_divergence = divergence.max()    
    return max_divergence, best_sigma

def vonNeumannEigenValues(Ev, lowRank = False, rank_proportion = 0.9):
    # rate: proportion of eigen values to keep
    # eigenvalues should be ordered descendengly 
    if lowRank:
        n_eig = int(rank_proportion*Ev.shape[0])
        Ev_lr = torch.zeros_like(Ev)
        Ev_lr[:n_eig] = Ev[:n_eig]
        Ev_lr[n_eig:] = torch.mean(Ev[n_eig:]) # make equal the last n_eig eigenvalues
        Ev_lr = Ev_lr / torch.sum(Ev_lr)
        H = -1*torch.sum(Ev_lr*torch.log(Ev_lr))
    else:
        mk = torch.gt(Ev, 0.0)
        mek = Ev[mk]
        mek = mek / torch.sum(mek)
        H = -1*torch.sum(mek*torch.log(mek))
    return H

def vonNeumannEntropyUnbiased(phiZ,sampling_size, n_permutations):
    _ , _, U = torch.linalg.svd(phiZ)
    Hperm = torch.zeros(n_permutations)
    for j in range(n_permutations):
        idxPerm = torch.randperm(len(phiZ))
        phiZperm = phiZ[idxPerm,:]
        phiZx = phiZperm[:sampling_size,:]
        phiZy = phiZperm[sampling_size:,:]
        
        projectionZx = phiZx@U.T
        projectionZy = phiZy@U.T
        ev_phiZx = (projectionZx**2).sum(dim = 0) # variance on each eigen vector
        ev_phiZy = (projectionZy**2).sum(dim = 0) # variance on each eigen vector
        # _ , S_sampled, U_sampled = torch.linalg.svd(phiZx)
        # ev_Real = S_sampled**2
        # print(ev_approx.shape)
        Hperm[j] = 0.5*(vonNeumannEigenValues(ev_phiZx) + vonNeumannEigenValues(ev_phiZy))
    
    H_expectation = torch.mean(Hperm)
    return H_expectation

def JSD_unbiased(X,Y,model, n_permutations):
    phiX = model(X)
    phiY = model(Y)

    covX = torch.matmul(torch.t(phiX),phiX)
    covY = torch.matmul(torch.t(phiY),phiY)

    Hx = vonNeumannEntropy(covX)
    Hy = vonNeumannEntropy(covY)
    phiZ =  torch.cat((phiX,phiY),dim = 0)

    Hz = vonNeumannEntropyUnbiased(phiZ,sampling_size = X.shape[0], n_permutations = n_permutations)
    jsd = Hz - 0.5*(Hx + Hy) # 
    return jsd



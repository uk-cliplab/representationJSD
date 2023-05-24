import numpy as np
import torch
import repitl.kernel_utils as ku
from ITL_utils import JSD_bound, QJSD, JSD_difference, deep_JSD, mmd, JSD_difference_augmented, JSD_difference_, JSD_unbiased
from models import DeepFourierFeatures, RFF_layer, DeepFourierFeaturesSingleKernel
from torch.utils.tensorboard import SummaryWriter
import datetime
from torch.optim.lr_scheduler import StepLR



def learningDeepRFF(X,Y,
    n_epochs = 500,
    d_in = 2,
    H = 50, # hidden units
    d_latent = 50,
    n_RFF = 20,
    learning_rate = 0.0005,
    is_cuda = 'True',
    typeKernel = 'product',
    variant = 'cosine_sine',
    freezeRFF = False,
    sigma_init = None,
    print_results  =False,
    print_every = 50,
    validation = False,
    Xval = None,
    Yval = None,
    saveLogs = False):

    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    # To track loss and parameters
    
    numSamples = X.shape[0]
    if sigma_init == None:
        d_in = X.shape[1]
        sigma_init = (2*d_in* torch.ones(1)) #as in deep MMD paper
    # elif initial_sigma == 'distance':
    #     sigma_init = 0.3*torch.sqrt(torch.sum(dists) / ( ((numSamples*2)**2 - (numSamples*2)) * 2 ))
    #     # print(sigma_init)
    # Locating the model and data in the corresponding device
    if is_cuda:
        device = torch.device('cuda:0')
        model = DeepFourierFeatures(d_in,H,d_latent, n_RFF, sigma_init,typeKernel , variant, freezeRFF).cuda()
        
    else:
        device = torch.device("cpu")
        model = DeepFourierFeatures(d_in,H,d_latent, n_RFF, sigma_init,typeKernel,variant,freezeRFF)
    X = X.to(device)
    Y = Y.to(device)
    if validation:
        Xval = Xval.to(device)
        Yval = Yval.to(device)

    # Defining optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) , lr = learning_rate)

    # scheduler = StepLR(optimizer, 
    #                step_size = 3000, # Period of learning rate decay
    #                gamma = 0.5) # Multiplicative factor of learning rate decay

    if validation:
        loss_val_max = 0
        best_model = DeepFourierFeatures(d_in,H,d_latent, n_RFF, sigma_init,typeKernel,variant,freezeRFF).cuda()
    if saveLogs:
        log_dir = "TST/higgs_n_" +str(numSamples) +"nRFF_" + str(n_RFF)+"sigm_" + str(sigma_init)+ "lr_"+str(learning_rate) +  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir)

    for epoch in range(n_epochs):


        # loss = -1*JSD_difference_(X, Y,model)
        loss = -1*deep_JSD(X,Y, model)
        optimizer.zero_grad()
        loss.backward() # retain_graph = True
        optimizer.step()
        # scheduler.step()
        # log loss and parameters
        # Validate model
        if validation:
            loss_val = validate_model(Xval, Yval, model)
            
            if -1*loss_val > loss_val_max:
                loss_val_max = -1*loss_val.detach()
                best_model.load_state_dict(model.state_dict())
        if saveLogs:
            loss_val = validate_model(Xval, Yval, model)
            loss_perm = validate_tst(Xval, Yval, model)
            writer.add_scalars('J(w)', {'training':-1*loss, 'validation': -1*loss_val, 'permutations': -1*loss_perm}, epoch)
            epsilon = torch.exp(model.eps.data) / (1 + torch.exp(model.eps.data))
            writer.add_scalar('eps', epsilon, epoch)




        if print_results:
            if epoch %print_every == 0:
                print("JSD: ", -1*loss.item() ) # -1*loss2.item()
                # print(psiX.to("cpu").detach())
    if validation:
        return best_model
    else:
        return model

def learningRFF(X,Y,typeDivergence = 'JSD', 
    n_RFF = 100, 
    freezeRFF = False, 
    alpha = 0.99, # in case of using renyi
    learning_rate = 0.001, 
    is_cuda = True,
    n_epochs = 50, 
    sigma_init = None, 
    validation = False,
    Xval = None, 
    Yval = None,
    saveLogs = False):

    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    # estimating an initial guess of sigma
    XY = torch.cat((X,Y))
    numSamples = X.shape[0] # + Y.shape[0]
    d_in = X.shape[1]
    

    # use this sigma_init for HDGM based on the dimension
    if sigma_init == None :
        sigma_init = (2*d_in * torch.rand([1])) ## as MMD-O

        
    # sigma = sigma_init.clone().detach()
    
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
    
    if validation:
        loss_val_max = 0
        best_model = RFF_layer(d_in, n_RFF,sigma_init,variant = 'cosine_sine',freezeRFF=freezeRFF).to(device)
        if saveLogs:
            log_dir = "TST/JSDRFF_n_" +str(numSamples) +"nRFF_" + str(n_RFF)+"sigm_" + str(sigma_init)+ "lr_"+str(learning_rate) +  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            writer = SummaryWriter(log_dir)
    

    
    for epoch in range(n_epochs):

        loss = -1*deep_JSD(X,Y,RFFestimator) 
   
        optimizer.zero_grad()
        loss.backward() # retain_graph = True
        optimizer.step()
        if validation:
            loss_val = validate_model(Xval, Yval, RFFestimator)
            
            if -1*loss_val > loss_val_max:
                loss_val_max = -1*loss_val.detach()
                best_model.load_state_dict(RFFestimator.state_dict())

            if saveLogs:
                loss_perm = validate_tst(Xval, Yval, RFFestimator)
                writer.add_scalars('J(w)', {'training':-1*loss, 'validation': -1*loss_val, 'permutations': -1*loss_perm}, epoch)

            
    if validation:
        return best_model
    else:
        return RFFestimator

def validate_model(Xval, Yval, model):
    loss = -1*deep_JSD(Xval, Yval, model)
    return loss

def validate_tst(Xval,Yval,model):
    # Creating the mixture of both distributions
    Zval =  torch.cat((Xval,Yval))

    # permuting the samples
    idxPerm = torch.randperm(len(Zval))
    Zperm = Zval[idxPerm,:]
    Xperm = Zperm[:Xval.shape[0],:]
    Yperm = Zperm[-Yval.shape[0]:,:]
    loss = -1*deep_JSD(Xperm, Yperm, model)
    return loss

def learningDeepRFFv2(X,Y,
    n_epochs = 500,
    d_in = 2,
    H = 50, # hidden units
    d_latent = 50,
    n_RFF = 20,
    learning_rate = 0.0005,
    is_cuda = 'True',
    typeKernel = 'product',
    variant = 'cosine_sine',
    freezeRFF = False,
    initial_sigma = 'distance',
    print_results  =False,
    print_every = 50,
    validation = False,
    Xval = None,
    Yval = None):

    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    # To track loss and parameters
    
    # estimating an initial guess of sigma
    XY = torch.cat((X,Y))
    numSamples = X.shape[0] # + Y.shape[0]
    dists = ku.squaredEuclideanDistance(XY, XY)
    if initial_sigma == 'dimension':
        d_in = X.shape[1]
        sigma_init = (2*d_in* torch.ones(1)) #as in deep MMD paper
    elif initial_sigma == 'distance':
        sigma_init = 0.5*torch.sqrt(torch.sum(dists) / ( ((numSamples*2)**2 - (numSamples*2)) * 2 ))
        # print(sigma_init)
    # Locating the model and data in the corresponding device
    if is_cuda:
        device = torch.device('cuda:0')
        model = DeepFourierFeatures(d_in,H,d_latent, n_RFF, sigma_init,typeKernel , variant, freezeRFF).cuda()
        
    else:
        device = torch.device("cpu")
        model = DeepFourierFeatures(d_in,H,d_latent, n_RFF, sigma_init,typeKernel,variant,freezeRFF)
    X = X.to(device)
    Y = Y.to(device)
    if validation:
        Xval = Xval.to(device)
        Yval = Yval.to(device)

    # Defining optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) , lr = learning_rate)

    # scheduler = StepLR(optimizer, 
    #                step_size = 3000, # Period of learning rate decay
    #                gamma = 0.5) # Multiplicative factor of learning rate decay

    if validation:
        log_dir = "TST/higgs/n_" +str(numSamples) +"nRFF_" + str(n_RFF)+"sigm_" + str(sigma_init)+ "lr_"+str(learning_rate) +  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir)

    for epoch in range(n_epochs):

        loss = -1*JSD_unbiased(X,Y,model, 20)

        optimizer.zero_grad()
        loss.backward() # retain_graph = True
        optimizer.step()
        # scheduler.step()
        # log loss and parameters
        # Validate model
        if validation:
            loss_val = validate_model(Xval, Yval, model)
            loss_perm = validate_tst(Xval, Yval, model)

            writer.add_scalars('J(w)', {'training':-1*loss, 'validation': -1*loss_val, 'permutations': -1*loss_perm}, epoch)
            epsilon = torch.exp(model.eps.data) / (1 + torch.exp(model.eps.data))
            writer.add_scalar('eps', epsilon, epoch)

        if print_results:
            if epoch %print_every == 0:
                print("JSD difference: ", -1*loss.item() ) # -1*loss2.item()
                # print(psiX.to("cpu").detach())
    
    return model

def bestSigmaDivergence(X,Y,
    n_epochs = 100,
    d_in = 2,
    H = 50, # hidden units
    d_latent = 50,
    n_RFF = 20,
    learning_rate = 0.0005,
    is_cuda = 'True',
    typeKernel = 'average',
    variant = 'cosine_sine',
    freezeRFF = False,
    print_every = 50):

    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)

    # estimating an initial guess of sigma
    XY = torch.cat((X,Y))
    numSamples = X.shape[0] # + Y.shape[0]
    dists = ku.squaredEuclideanDistance(XY, XY)
    sigma_init = torch.sqrt(torch.sum(dists) / ( ((numSamples*2)**2 - (numSamples*2)) * 2 ))
    # Locating the model and data in the corresponding device
    if is_cuda:
        device = torch.device('cuda:0')
        model = DeepFourierFeatures(d_in,H,d_latent, n_RFF, sigma_init,typeKernel , variant, freezeRFF).cuda()
        
    else:
        device = torch.device("cpu")
        model = DeepFourierFeatures(d_in,H,d_latent, n_RFF, sigma_init,typeKernel,variant,freezeRFF)
    X = X.to(device)
    Y = Y.to(device)

    # Defining optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) , lr = learning_rate)

    for epoch in range(n_epochs):
        # to regularize epsilon so that is always between 0 and 1
        # eps_ = torch.exp(eps) / (1 + torch.exp(eps))
        # COmpute mappings of the product kernel (1-eps)*K(phiw(x),phiw(x'))*q(x,x') + eps*q(x,x')
        phiX = model(X)
        phiY = model(Y)
        

        # Creating the mixture of both distributions
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
        
        # estimating the difference between the JSD of the real distributions and the permutations
        loss = -1*JSD_difference(covX, covY,covXperm,covYperm)
        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()
        
        if epoch %print_every == 0:
            print("JSD difference: ", -1*loss.item() ) # -1*loss2.item()
            # print(psiX.to("cpu").detach())
    
    return model


def RFFlearning(X,Y,
    n_epochs = 500,
    d_in = 2,
    n_RFF = 20,
    learning_rate = 0.0005,
    is_cuda = 'True',
    variant = 'cosine_sine',
    print_every = 50):

    # estimating an initial guess of sigma
    XY = torch.cat((X,Y))
    numSamples = X.shape[0] # + Y.shape[0]
    dists = ku.squaredEuclideanDistance(XY, XY)
    sigma_init = torch.sqrt(torch.sum(dists) / ( ((numSamples*2)**2 - (numSamples*2)) * 2 ))
    # Locating the model and data in the corresponding device
    if is_cuda:
        device = torch.device('cuda:0')
        model =  RFF_layer(d_in, d_out =n_RFF,sigma_init = sigma_init,variant = variant,freezeRFF=False).cuda()

        
    else:
        device = torch.device("cpu")
        model =  RFF_layer(d_in, d_out =n_RFF,sigma_init = sigma_init,variant = variant,freezeRFF=False).cuda()
    X = X.to(device)
    Y = Y.to(device)

    # Defining optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) , lr = learning_rate)

    for epoch in range(n_epochs):
        # to regularize epsilon so that is always between 0 and 1
        # eps_ = torch.exp(eps) / (1 + torch.exp(eps))
        # COmpute mappings of the product kernel (1-eps)*K(phiw(x),phiw(x'))*q(x,x') + eps*q(x,x')
        phiX = model(X)
        phiY = model(Y)
        

        # Creating the mixture of both distributions
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
        
        # estimating the difference between the JSD of the real distributions and the permutations
        loss = -1*JSD_difference(covX, covY,covXperm,covYperm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch %print_every == 0:
            print("JSD difference: ", -1*loss.item() ) # -1*loss2.item()
            # print(psiX.to("cpu").detach())
    
    return model

def deepRFFsingleKernel(X,Y,
    n_epochs = 1000,
    d_in = 2,
    H = 50,
    d_latent = 50,
    n_RFF = 20,
    learning_rate = 0.0005,
    is_cuda = 'True',
    variant = 'cosine_sine',
    freezeRFF = False,
    print_every = 50):

    # estimating an initial guess of sigma
    XY = torch.cat((X,Y))
    numSamples = X.shape[0] # + Y.shape[0]
    dists = ku.squaredEuclideanDistance(XY, XY)
    sigma_init = torch.sqrt(torch.sum(dists) / ( ((numSamples*2)**2 - (numSamples*2)) * 2 ))
    # Locating the model and data in the corresponding device
    if is_cuda:
        device = torch.device('cuda:0')
        model =  DeepFourierFeaturesSingleKernel(d_in, H, d_latent, n_RFF,sigma_init,variant,freezeRFF).cuda()

        
    else:
        device = torch.device("cpu")
        model =  DeepFourierFeaturesSingleKernel(d_in, H, d_latent, n_RFF,sigma_init,variant,freezeRFF).cuda()
    X = X.to(device)
    Y = Y.to(device)

    # Defining optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) , lr = learning_rate)

    for epoch in range(n_epochs):
        # to regularize epsilon so that is always between 0 and 1
        # eps_ = torch.exp(eps) / (1 + torch.exp(eps))
        # COmpute mappings of the product kernel (1-eps)*K(phiw(x),phiw(x'))*q(x,x') + eps*q(x,x')
        phiX = model(X)
        phiY = model(Y)
        

        # Creating the mixture of both distributions
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
        
        # estimating the difference between the JSD of the real distributions and the permutations
        loss = -1*JSD_difference(covX, covY,covXperm,covYperm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch %print_every == 0:
            print("JSD difference: ", -1*loss.item() ) # -1*loss2.item()
            # print(psiX.to("cpu").detach())
    
    return model
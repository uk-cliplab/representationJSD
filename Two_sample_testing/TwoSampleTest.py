
import numpy as np
import torch
import torch.utils.data
import repitl.kernel_utils as ku
from ITL_utils import  QJSD, deep_JSD, mmd, optimizedDivergence, bestSigmaDivergence, bestSigmaDivergenceRFF
from utils import sample_blobs,HDGM, Higgs
from trainDeepRff import learningDeepRFF, learningRFF
from baselines import train_MMD_O, TST_MMD_adaptive_bandwidth, train_deep_MMD, TST_MMD_u, TST_MMD_u_HD, fitC2ST, C2ST_perm_test


# default `log_dir` is "runs" - we'll be more specific here


def deep_JSD_permutation_test(X, Y, model, significance, rep):
    """
    Performs a permutation test on the JSD method
    """ 
    N_X = len(X)
    N_Y = len(Y)
    Z = torch.cat([X, Y], dim=0)
    jsd = []
    
    for i in range(rep):
        Z = Z[torch.randperm(len(Z)),:]
        div = deep_JSD(Z[:N_X, :], Z[N_X:, :],model)
        jsd.append(div)

        
    jsd = torch.tensor(jsd)
    jsd = jsd.to(X.device)
    thr_jsd = torch.quantile(jsd, (1 - significance))
    return thr_jsd
    
def jsd_permutation_test(x, y, sigma, significance, rep):
    """
    Performs a permutation test on the JSD method
    """
    
    N_X = len(x)
    N_Y = len(y)
    xy = torch.cat([x, y], dim=0).double()
    jsd = []
    
    for i in range(rep):
        xy = xy[torch.randperm(len(xy))]
        div = QJSD(xy[:N_X, :], xy[N_X:, :],sigma = sigma)
        jsd.append(div)
        
    jsd = torch.tensor(jsd)
    thr_jsd = torch.quantile(jsd, (1 - significance))
    return thr_jsd

# def baselines_permutation_test(samples_per_blob,rep,device):





def mmd_permutation_test(x, y, sigma, significance, rep):
    """
    Performs a permutation test on the MMD method
    """
    
    N_X = len(x)
    N_Y = len(y)
    xy = torch.cat([x, y], dim=0).double()
    mmds = []
    
    for i in range(rep):
        xy = xy[torch.randperm(len(xy))]
        # div = optimizedDivergence(xy[:N_X, :],xy[N_X:, :],typeDivergence = 'MMD')
        # mmds.append(div)
        mmds.append(mmd(xy[:N_X, :], xy[N_X:, :], sigma).item())
        
    mmds = torch.tensor(mmds)
    thr_mmd = torch.quantile(mmds, (1 - significance))
    return thr_mmd


def run_experiment_blobs(samples_per_blob,
    numRepetitions, 
    numTestSets,
    permTestSize,    
    significance, 
    is_cuda = True,
    parallel = False,
    pId = 0, 
    ):
    """
    Generalized form of the mean/variance experiment
    
    Inputs:
        dataGenFunction: A function that 
                         
        iterableDiffs: A list of differnet values to iterate over. Can be differences in means or variances.
                       Items inside are passed to the dataGenFunction to create data
    """
    # Parameters definition for baselines
    
    d_in = 2
    d_latent = 50
    num_neurons_c2st = 256
    d_out = 50
    dtype = torch.float
    learning_rate_MMD_O = 0.0005
    learning_rate_MMD_D = 0.0005
    learning_rate_jsd = 0.01
    learning_rate_jsd_rff = 0.001 # best for jsd_ff
    learning_rate_djsd = 0.001 # 0.005 best for JSD-D for higher samples, maybe a smaller one will work better for a few samples
    learning_rate_c = 1e-4
    N_epoch = 1000
    epochs_c2st =  200
    batch_size_c2st = 36

    n_RFF = 50 
     
    # Setup seeds
    np.random.seed(1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    torch.backends.cudnn.deterministic = False

    if is_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device("cpu")
        
    # storage for results
    
    jsd_results = np.zeros([numRepetitions, numTestSets,len(samples_per_blob)])
    jsd_results_ff = np.zeros([numRepetitions, numTestSets, len(samples_per_blob)])  
    jsd_results_rff = np.zeros([numRepetitions, numTestSets, len(samples_per_blob)]) 
    deep_jsd_results = np.zeros([numRepetitions, numTestSets, len(samples_per_blob)])
    deep_mmd_results = np.zeros([numRepetitions, numTestSets, len(samples_per_blob)])
    mmd_results = np.zeros([numRepetitions, numTestSets, len(samples_per_blob)])
    c2st_s_results = np.zeros([numRepetitions, numTestSets, len(samples_per_blob)])
    c2st_l_results = np.zeros([numRepetitions, numTestSets, len(samples_per_blob)])

    if parallel:
        IterableRep = [pId-1,] # the the process Id
    else:
        IterableRep = range(numRepetitions)
    
    # main experiment loop
    for repIdx in IterableRep:
        
        for sIdx, samples in enumerate(samples_per_blob):
            print("rep: ", repIdx +1, ", sIdx: ", sIdx +1)
            seed_init = 1102*repIdx      
            torch.manual_seed(repIdx * 19 + samples)
            torch.cuda.manual_seed(repIdx * 19 + samples)

            N_total = 9*samples
            # construct two distributions
            Xtrain, Ytrain = sample_blobs(samples, rs = seed_init)
            Xval, Yval = sample_blobs(samples, rs = seed_init + 1 )
        
            # collect baselines 
            # Train C2ST-L
            S = torch.cat([Xtrain,Ytrain], dim = 0).to(device)
            model_c = fitC2ST(Xtrain, Ytrain, d_in,num_neurons_c2st,2, batch_size_c2st, learning_rate_c, epochs_c2st, device)           
            # Train MMD - O
            sigma0 = train_MMD_O(S, N_total, learning_rate_MMD_O, N_epoch)
            # Train Deep MMD 
            model_u, sigma, sigma0_u, ep = train_deep_MMD(S, N_total, d_in, d_latent, d_out, learning_rate_MMD_D, N_epoch, device, dtype, initial_sigma = 'random')
            # Learn sigma for JSD directly from the gram-matrix
            ##### ----- Rule of thumb to guess of initial sigma ---------###
            XY = torch.cat((Xtrain,Ytrain))
            numSamples = Xtrain.shape[0] # + Y.shape[0]
            dists = ku.squaredEuclideanDistance(XY, XY)
            sigma_init = torch.sqrt(torch.sum(dists) / ( ((numSamples*2)**2 - (numSamples*2)) * 2 ))
            ## -----------------------------------------------------------------------------------------##
            sigma_jsd = bestSigmaDivergence(Xtrain,Ytrain,typeDivergence = 'JSD', learning_rate = learning_rate_jsd, is_cuda = is_cuda,n_epochs = 100, initial_sigma = 'distance', validation = False, Xval = Xval, Yval = Yval) # hundred epochs are enough here
            # learn the FF to approximate JSD for some n_RFF


            model_jsd_rff0 = learningRFF(Xtrain,Ytrain,typeDivergence = 'JSD', n_RFF = n_RFF,freezeRFF = True, learning_rate = learning_rate_jsd_rff, is_cuda = True,n_epochs = N_epoch, sigma_init=0.2*sigma_init) # this one requieres more epochs
            model_jsd_ff0 = learningRFF(Xtrain,Ytrain,typeDivergence = 'JSD', n_RFF = n_RFF, freezeRFF = False, learning_rate = learning_rate_jsd_rff, is_cuda = True,n_epochs = N_epoch,  sigma_init=0.2*sigma_init, validation = False, Xval= None, Yval = None ) # this one requieres more epochs

            # _ , sigma_jsd = optimizedDivergence(Xtrain,Ytrain,typeDivergence = 'JSD') # using secant algorithm
            # Learn the kernel for deep JSD with different n_RFF
            model0 = learningDeepRFF(Xtrain,Ytrain, typeKernel='product',n_epochs = N_epoch, n_RFF=n_RFF, learning_rate = learning_rate_djsd, is_cuda=is_cuda, freezeRFF = False, sigma_init=0.2*sigma_init, validation=False, Xval= Xval, Yval = Yval)
            

            ## TESTING THE LEARNED KERNELS AND MODEL ON TESTING SETS
            for testIdx in range(numTestSets):
                torch.cuda.empty_cache()
                # Use testing set to get the estimate and then compare to the distribution of permutations
                seed_testing = 1102 * (testIdx+2) + 2*repIdx
                Xtest, Ytest = sample_blobs(samples, rs = seed_testing)

                # Compute thresholds and power test
                # MMD- O
                Stest = torch.cat([Xtest,Ytest], dim = 0).to(device)
                h_adaptive, _, _ = TST_MMD_adaptive_bandwidth(Stest, permTestSize, N_total, Stest, 0, sigma0, significance, device, dtype)
                mmd_results[repIdx,testIdx,sIdx] = h_adaptive
                # Deep MMD
                h_u, _, _ = TST_MMD_u(model_u(Stest), permTestSize, N_total, Stest, sigma, sigma0_u, significance, device,
                                                dtype, ep)
                deep_mmd_results[repIdx,testIdx,sIdx] = h_u
                
                # C2ST-S
                c2st_s_results[repIdx,testIdx,sIdx] = C2ST_perm_test(Xtest,Ytest, model_c, permTestSize, significance, 's', device)    
                
                # C2ST-L
                c2st_l_results[repIdx,testIdx,sIdx] = C2ST_perm_test(Xtest,Ytest, model_c, permTestSize, significance, 'l',device)

                # collect JSD baseline and permutation test
                jsd_ = QJSD(Xtest,Ytest, sigma = sigma_jsd)
                thr_jsd = jsd_permutation_test(Xtest, Ytest, sigma_jsd, significance, permTestSize)
                if thr_jsd < jsd_:
                    jsd_results[repIdx,testIdx,sIdx] = 1   ## indexing in the fourth position the original jsd to preserve the order
                # Now we do it for the model with Forier Features (we can use the same function of deep jsd because theres a network that computes the mapping explicitely)
                jsd_rff_ = deep_JSD(Xtest.to(device),Ytest.to(device),model_jsd_rff0)
                thr_jsd_rff = deep_JSD_permutation_test(Xtest.to(device), Ytest.to(device), model_jsd_rff0, significance, permTestSize)
                jsd_ff_ = deep_JSD(Xtest.to(device),Ytest.to(device),model_jsd_ff0)
                thr_jsd_ff = deep_JSD_permutation_test(Xtest.to(device), Ytest.to(device), model_jsd_ff0, significance, permTestSize)



                if thr_jsd_rff < jsd_rff_:
                    jsd_results_rff[repIdx,testIdx,sIdx] = 1
                if thr_jsd_ff < jsd_ff_:
                    jsd_results_ff[repIdx,testIdx,sIdx] = 1
   

                # Now, for the deep JSD models              
                deep_jsd_0 = deep_JSD(Xtest.to(device),Ytest.to(device),model0)
                thr_deep_jsd0 = deep_JSD_permutation_test(Xtest.to(device), Ytest.to(device), model0, significance, permTestSize)

                
                if thr_deep_jsd0 < deep_jsd_0:
                    deep_jsd_results[repIdx,testIdx,sIdx] = 1
                           
    return jsd_results, jsd_results_rff, jsd_results_ff, deep_jsd_results, mmd_results, deep_mmd_results, c2st_s_results, c2st_l_results

def run_experiment_HDGM(samples_per_cluster,
    dimension,
    numRepetitions, 
    numTestSets,
    permTestSize,    
    significance, 
    is_cuda = True,
    parallel = False,
    pId = 0):
    """
    """
    # Parameters definition for baselines
    

    dtype = torch.float
    learning_rate_MMD_O = 0.001 # following paper parameters (https://github.com/fengliu90/DK-for-TST/blob/master/Deep_Kernel_HDGM.py)
    learning_rate_MMD_D = 0.00005 # Following paper parameters
    learning_rate_djsd = 0.005 # 0.005
    learning_rate_jsd_rff = 0.05 
    learning_rate_c = 1e-4
    N_epoch = 1000
    N_epoch_jsd = 200
    epochs_c2st =  200
    batch_size_c2st = 128

    # Setup seeds
    np.random.seed(1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    torch.backends.cudnn.deterministic = True
     

    if is_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device("cpu")
        
    # storage for results
    # jsd_results = np.zeros([numRepetitions, numTestSets, len(samples_per_cluster), len(dimension)])   
    jsd_results_rff  = np.zeros([numRepetitions, numTestSets, len(samples_per_cluster), len(dimension)]) 
    jsd_results_ff   = np.zeros([numRepetitions, numTestSets, len(samples_per_cluster), len(dimension)])
    deep_jsd_results = np.zeros([numRepetitions, numTestSets, len(samples_per_cluster),len(dimension)])
    deep_mmd_results = np.zeros([numRepetitions, numTestSets, len(samples_per_cluster),len(dimension)])
    mmd_results      = np.zeros([numRepetitions, numTestSets, len(samples_per_cluster),len(dimension)])
    c2st_s_results   = np.zeros([numRepetitions, numTestSets, len(samples_per_cluster),len(dimension)])
    c2st_l_results   = np.zeros([numRepetitions, numTestSets, len(samples_per_cluster),len(dimension)])

    if parallel:
        IterableRep = [pId-1,]
    else:
        IterableRep = range(numRepetitions)
    
    # main experiment loop
    for repIdx in IterableRep:
        for didx, d in enumerate(dimension):
        
            for sIdx, samples in enumerate(samples_per_cluster):

                print("rep: ", repIdx +1, ", sIdx: ", sIdx +1, ', dIdx: ', didx +1)
                seed_init = 1102*repIdx      
                torch.manual_seed(repIdx * 19 + samples)
                torch.cuda.manual_seed(repIdx * 19 + samples)
                d_in = d
                H = 3*d ## hidden units or neurons
                d_out = 3*d
                num_neurons_c2st = 3*d # to be fair all models are very similar in architecture ..
                N_total = 2*samples # samples = samples per cluster
                n_RFF = 15 # int(0.5*(np.sqrt(N_total)*np.log(N_total))) # 
                # construct two distributions
                Xtrain, Ytrain = HDGM(samples, d,seed_init)
                Xval, Yval = HDGM(samples, d, seed_init +1 )

            
                # collect baselines and permutation test
                # Train C2ST-L
                S = torch.cat([Xtrain,Ytrain], dim = 0).to(device)
                
                model_c = fitC2ST(Xtrain, Ytrain, d_in,num_neurons_c2st,2, batch_size_c2st, learning_rate_c, epochs_c2st, device)    # 2 number of classes       
                # Train MMD - O
                
                sigma0 = train_MMD_O(S, N_total, learning_rate_MMD_O, N_epoch, initial_sigma = 'dimension')
                # Train Deep MMD 
                
                model_u, sigma, sigma0_u, ep = train_deep_MMD(S, N_total, d_in, H, d_out, learning_rate_MMD_D, N_epoch, device, dtype, initial_sigma = 'dimension', validation = False, Xval =Xval, Yval = Yval)
               
                # Learn sigma for JSD directly from the gram-matrix
                # sigma_jsd = bestSigmaDivergence(Xtrain,Ytrain,typeDivergence = 'JSD', learning_rate = learning_rate_jsd, is_cuda = is_cuda,n_epochs = 100, initial_sigma = 'dimension', validation = False, Xval = Xval, Yval = Yval) # hundred epochs are enough here
                # learn the FF to approximate JSD for some n_RFF

                ##### ----- Rule of thumb to guess of initial sigma ---------###
                XY = torch.cat((Xtrain,Ytrain))
                numSamples = Xtrain.shape[0] # + Y.shape[0]
                dists = ku.squaredEuclideanDistance(XY, XY)
                sigma_init = torch.sqrt(torch.sum(dists) / ( ((numSamples*2)**2 - (numSamples*2)) * 2 ))

                sigma_init_d = (2*d_in* torch.ones(1))
                ##------------------------------------------------------------##
                
                model_jsd_rff0 = learningRFF(Xtrain,Ytrain,typeDivergence = 'JSD', n_RFF = n_RFF, freezeRFF = True, learning_rate = learning_rate_jsd_rff, is_cuda = is_cuda,n_epochs = N_epoch_jsd, sigma_init=2*sigma_init,validation= True, Xval=Xval, Yval = Yval) 

                model_jsd_ff0 = learningRFF(Xtrain,Ytrain,typeDivergence = 'JSD', n_RFF = n_RFF, freezeRFF = False, learning_rate = learning_rate_jsd_rff, is_cuda = True,n_epochs = N_epoch_jsd, sigma_init=2*sigma_init,validation= True, Xval=Xval, Yval = Yval) # this one requieres more epochs

                # _ , sigma_jsd = optimizedDivergence(Xtrain,Ytrain,typeDivergence = 'JSD') # using secant algorithm
                # Learn the kernel for deep JSD with different n_RFF
                
                model0 = learningDeepRFF(Xtrain,Ytrain, H = H, d_latent = d_out, n_RFF=n_RFF, learning_rate = learning_rate_djsd,typeKernel='product',n_epochs = N_epoch_jsd, d_in = d_in , sigma_init=2*sigma_init, is_cuda=is_cuda, freezeRFF = False, validation= True, Xval=Xval, Yval = Yval)

                # model_augmented = learningDeepRFF(Xtrain,Ytrain, H = H, d_latent = d_out, n_RFF=n_RFF, learning_rate = learning_rate_djsd,typeKernel='product',n_epochs = N_epoch, d_in= d_in , initial_sigma = 'dimension', is_cuda=is_cuda, freezeRFF = False, augmented = True, validation= True, Xval=Xval, Yval = Yval)

                ## TESTING THE LEARNED KERNELS AND MODEL ON TESTING SETS
                for testIdx in range(numTestSets):
                    # Use testing set to get the estimate and then compare to the distribution of permutations
                    seed_testing = 1102 * (testIdx+2) + 2*repIdx
                    Xtest, Ytest = HDGM(samples, d, seed_testing)

                    # Compute thresholds and power test
                    # MMD- O
                    Stest = torch.cat([Xtest,Ytest], dim = 0).to(device)
                    h_adaptive, _, _ = TST_MMD_adaptive_bandwidth(Stest, permTestSize, N_total, Stest, 0, sigma0, significance, device, dtype)
                    mmd_results[repIdx,testIdx,sIdx,didx] = h_adaptive
                    # Deep MMD
                    h_u, _, _ = TST_MMD_u_HD(model_u(Stest), permTestSize, N_total, Stest, sigma, sigma0_u, ep, significance, device,
                                                    dtype, ep)
                    deep_mmd_results[repIdx,testIdx,sIdx,didx] = h_u
                    
                    # C2ST-S
                    c2st_s_results[repIdx,testIdx,sIdx,didx] = C2ST_perm_test(Xtest,Ytest, model_c, permTestSize, significance, 's', device)    
                    
                    # C2ST-L
                    c2st_l_results[repIdx,testIdx,sIdx,didx] = C2ST_perm_test(Xtest,Ytest, model_c, permTestSize, significance, 'l',device)

                    # # # collect JSD baseline and permutation test
                    # # collect JSD baseline and permutation test
                    # jsd_ = QJSD(Xtest,Ytest, sigma = sigma_jsd)
                    # thr_jsd = jsd_permutation_test(Xtest, Ytest, sigma_jsd, significance, permTestSize)
                    # if thr_jsd < jsd_:
                    #     jsd_results[repIdx,testIdx,sIdx,didx] = 1   ## indexing in the fourth position the original jsd to preserve the order


                    
                    jsd_rff_ = deep_JSD(Xtest,Ytest,model_jsd_rff0.to("cpu"))
                    thr_jsd_rff = deep_JSD_permutation_test(Xtest, Ytest, model_jsd_rff0.to("cpu"), significance, permTestSize)
                    jsd_ff_ = deep_JSD(Xtest,Ytest,model_jsd_ff0.to("cpu"))
                    thr_jsd_ff = deep_JSD_permutation_test(Xtest, Ytest, model_jsd_ff0.to("cpu"), significance, permTestSize)


                    if thr_jsd_rff < jsd_rff_:
                        jsd_results_rff[repIdx,testIdx,sIdx,didx] = 1
                    if thr_jsd_ff < jsd_ff_:
                        jsd_results_ff[repIdx,testIdx,sIdx,didx] = 1
    

                    # Now, for the deep JSD models              
                    deep_jsd_0 = deep_JSD(Xtest,Ytest,model0.to("cpu"))
                    thr_deep_jsd0 = deep_JSD_permutation_test(Xtest, Ytest, model0.to("cpu"), significance, permTestSize)

                    if thr_deep_jsd0 < deep_jsd_0:
                        deep_jsd_results[repIdx,testIdx,sIdx,didx] = 1
                                             
    return jsd_results_rff, jsd_results_ff, deep_jsd_results, mmd_results, deep_mmd_results, c2st_s_results, c2st_l_results

def run_experiment_Higgs(n_samples,
    numRepetitions, 
    numTestSets,
    permTestSize,    
    significance, 
    is_cuda = True,
    parallel = False,
    pId = 1):
    """
    Generalized form of the mean/variance experiment
    
    Inputs:
        dataGenFunction: A function that 
                         
        iterableDiffs: A list of differnet values to iterate over. Can be differences in means or variances.
                       Items inside are passed to the dataGenFunction to create data
    """
    # Parameters definition for baselines
    

    dtype = torch.float
    learning_rate_MMD_O = 0.001 # following paper parameters (https://github.com/fengliu90/DK-for-TST/blob/master/Deep_Kernel_HDGM.py)
    learning_rate_MMD_D = 0.00005 # Following paper parameters
    learning_rate_djsd = 0.01
    learning_rate_jsd_rff = 0.01
    # learning_rate_djsd = 0.05
    # learning_rate_jsd_rff = 0.05
    learning_rate_c = 1e-4
    N_epoch = 1000
    epochs_c2st =  200
    batch_size_c2st = 128

    # Setup seeds
    np.random.seed(1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    torch.backends.cudnn.deterministic = True
     

    if is_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device("cpu")
        
    # storage for results
    

    jsd_results_ff = np.zeros([numRepetitions, numTestSets, len(n_samples)])  
    jsd_results_rff = np.zeros([numRepetitions, numTestSets, len(n_samples)])  
    deep_jsd_results = np.zeros([numRepetitions, numTestSets, len(n_samples)])
    deep_mmd_results = np.zeros([numRepetitions, numTestSets, len(n_samples)])
    mmd_results = np.zeros([numRepetitions, numTestSets, len(n_samples)])
    c2st_s_results = np.zeros([numRepetitions, numTestSets, len(n_samples)])
    c2st_l_results = np.zeros([numRepetitions, numTestSets, len(n_samples)])

    if parallel:
        IterableRep = [pId-1,]
    else:
        IterableRep = range(numRepetitions)
    
    # main experiment loop
    for repIdx in IterableRep:   
        for sIdx, samples in enumerate(n_samples):

            print("rep: ", repIdx +1, ", sIdx: ", sIdx +1)
            seed_init = 1102*repIdx      
            torch.manual_seed(repIdx * 19 + samples)
            torch.cuda.manual_seed(repIdx * 19 + samples)
            d_in = 4 # HIGGS dataset 4 dimensional
            H = 20 ## hidden units or neurons
            d_out = 20
            num_neurons_c2st = 20 # to be fair all models are very similar in architecture ..
            N_total = samples 
            n_RFF =  15 #int(0.5*(np.sqrt(N_total)*np.log(N_total))) # 
            print("number of Fourier Features: " ,n_RFF)
            # construct two distributions
            Xtrain, Ytrain = Higgs(samples,seed_init)
            Xval, Yval = Higgs(samples,seed_init+ 1)
            
        
            # collect baselines and permutation test
            # Train C2ST-L
            S = torch.cat([Xtrain,Ytrain], dim = 0).to(device)
            
            model_c = fitC2ST(Xtrain, Ytrain, d_in,num_neurons_c2st,2, batch_size_c2st, learning_rate_c, epochs_c2st, device)    # 2 number of classes       
            # Train MMD - O
            
            sigma0 = train_MMD_O(S, N_total, learning_rate_MMD_O, N_epoch,initial_sigma = 'dimension')
            # Train Deep MMD 
            
            model_u, sigma, sigma0_u, ep = train_deep_MMD(S, N_total, d_in, H, d_out, learning_rate_MMD_D, N_epoch, device, dtype, initial_sigma = 'higgs')
            # Learn sigma for JSD directly from the gram-matrix  (too expensive for this experiment)
            # sigma_jsd = bestSigmaDivergence(Xtrain,Ytrain,typeDivergence = 'JSD', learning_rate = learning_rate_jsd, is_cuda = is_cuda,n_epochs = 100) # hundred epochs are enough here
            # learn the FF to approximate JSD for some n_RFF

            ##### ----- Rule of thumb to guess of initial sigma ---------###
            XY = torch.cat((Xtrain,Ytrain))
            numSamples = Xtrain.shape[0] # + Y.shape[0]
            dists = ku.squaredEuclideanDistance(XY, XY)
            sigma_init = torch.sqrt(torch.sum(dists) / ( ((numSamples*2)**2 - (numSamples*2)) * 2 ))

            sigma_init_d = (2*d_in* torch.ones(1))
            ##------------------------------------------------------------##
            
            model_jsd_rff0 = learningRFF(Xtrain,Ytrain,typeDivergence = 'JSD', n_RFF = n_RFF, freezeRFF = True, learning_rate = learning_rate_jsd_rff, is_cuda = is_cuda,n_epochs = N_epoch, sigma_init=1*sigma_init,validation= True, Xval=Xval, Yval = Yval) # this one requieres more epochs
            model_jsd_ff0 = learningRFF(Xtrain,Ytrain,typeDivergence = 'JSD', n_RFF = n_RFF, freezeRFF = False, learning_rate = learning_rate_jsd_rff, is_cuda = is_cuda,n_epochs = N_epoch, sigma_init=1*sigma_init,validation= True, Xval=Xval, Yval = Yval) # this one requieres more epochs

            # _ , sigma_jsd = optimizedDivergence(Xtrain,Ytrain,typeDivergence = 'JSD') # using secant algorithm
            # Learn the kernel for deep JSD with different n_RFF
            
            model0 = learningDeepRFF(Xtrain,Ytrain, H = H, d_latent = d_out, n_RFF=n_RFF, learning_rate = learning_rate_djsd, typeKernel='product',n_epochs = N_epoch, d_in= d_in , is_cuda=is_cuda,sigma_init=1*sigma_init,freezeRFF = False, validation= True, Xval=Xval, Yval = Yval)


            ## TESTING THE LEARNED KERNELS AND MODEL ON TESTING SETS
            for testIdx in range(numTestSets):
                # Use testing set to get the estimate and then compare to the distribution of permutations
                seed_testing = 1102 * (testIdx+2) + 2*repIdx
                Xtest, Ytest = Higgs(samples, seed_testing)

                # Compute thresholds and power test
                # MMD- O
                Stest = torch.cat([Xtest,Ytest], dim = 0).to(device)
                h_adaptive, _, _ = TST_MMD_adaptive_bandwidth(Stest, permTestSize, N_total, Stest, 0, sigma0, significance, device, dtype)
                mmd_results[repIdx,testIdx,sIdx] = h_adaptive
                # Deep MMD
                h_u, _, _ = TST_MMD_u_HD(model_u(Stest), permTestSize, N_total, Stest, sigma, sigma0_u, ep, significance, device,
                                                dtype, ep)
                deep_mmd_results[repIdx,testIdx,sIdx] = h_u
                
                # C2ST-S
                c2st_s_results[repIdx,testIdx,sIdx] = C2ST_perm_test(Xtest,Ytest, model_c, permTestSize, significance, 's', device)    
                
                # C2ST-L
                c2st_l_results[repIdx,testIdx,sIdx] = C2ST_perm_test(Xtest,Ytest, model_c, permTestSize, significance, 'l',device)

                # # collect JSD baseline and permutation test
                # jsd_ = QJSD(Xtest,Ytest, sigma = sigma_jsd)
                # thr_jsd = jsd_permutation_test(Xtest, Ytest, sigma_jsd, significance, permTestSize)
                # Now we do it for the model with Forier Features (we can use the same function of deep jsd because theres a network that computes the mapping explicitely)
                jsd_rff_ = deep_JSD(Xtest,Ytest,model_jsd_rff0.to("cpu"))
                thr_jsd_rff = deep_JSD_permutation_test(Xtest, Ytest, model_jsd_rff0.to("cpu"), significance, permTestSize)
                jsd_ff_ = deep_JSD(Xtest,Ytest,model_jsd_ff0.to("cpu"))
                thr_jsd_ff = deep_JSD_permutation_test(Xtest, Ytest, model_jsd_ff0.to("cpu"), significance, permTestSize)

                if thr_jsd_rff < jsd_rff_:
                    jsd_results_rff[repIdx,testIdx,sIdx] = 1
                if thr_jsd_ff < jsd_ff_:
                    jsd_results_ff[repIdx,testIdx,sIdx] = 1


                # Now, for the deep JSD models              
                deep_jsd_0 = deep_JSD(Xtest,Ytest,model0.to("cpu"))
                thr_deep_jsd0 = deep_JSD_permutation_test(Xtest, Ytest, model0.to("cpu"), significance, permTestSize)

                if thr_deep_jsd0 < deep_jsd_0:
                    deep_jsd_results[repIdx,testIdx,sIdx] = 1
                    
                    
                           
    return jsd_results_rff, jsd_results_ff, deep_jsd_results, mmd_results, deep_mmd_results, c2st_s_results, c2st_l_results

def run_experiment_baselines_rff(samples_per_blob,
    numRepetitions, 
    numTestSets,
    permTestSize,    
    significance, 
    is_cuda = True):
    """
    Generalized form of the mean/variance experiment
    
    Inputs:
        dataGenFunction: A function that creates gaussian data with a specific mean or variance.
                         This data generation function is specific to the actual experiment, such as means differences experiment.
                         
        iterableDiffs: A list of differnet values to iterate over. Can be differences in means or variances.
                       Items inside are passed to the dataGenFunction to create data
    """
    # Parameters definition for baselines
    d_in = 2
    d_latent = 50
    num_neurons_c2st = 256
    d_out = 50
    dtype = torch.float
    learning_rate_MMD_O = 0.0005
    learning_rate_MMD_D = 0.0005
    learning_rate_jsd = 0.01
    learning_rate_jsd_rff = 0.0005
    learning_rate_c = 1e-4
    N_epoch = 1000
    epochs_c2st =  200
    batch_size_c2st = 36

    n_RFF = [10,50,100,500] # different configurations of RFF that we are going to try
     

    if is_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device("cpu")
        
    # storage for results
    
    jsd_results = np.zeros([numRepetitions, numTestSets, len(n_RFF)+1, len(samples_per_blob)]) # including results with the kernel itself
    deep_jsd_results = np.zeros([numRepetitions, numTestSets,  len(n_RFF), len(samples_per_blob)])
    deep_mmd_results = np.zeros([numRepetitions, numTestSets, len(samples_per_blob)])
    mmd_results = np.zeros([numRepetitions, numTestSets, len(samples_per_blob)])
    c2st_s_results = np.zeros([numRepetitions, numTestSets, len(samples_per_blob)])
    c2st_l_results = np.zeros([numRepetitions, numTestSets, len(samples_per_blob)])


    
    # main experiment loop
    for repIdx in range(numRepetitions):
        
        for sIdx, samples in enumerate(samples_per_blob):
            print("rep: ", repIdx +1, ", sIdx: ", sIdx +1)

            N_total = 9*samples
            # construct two distributions
            Xtrain, Ytrain = sample_blobs(samples)
        
            # collect baselines and permutation test
            # Train C2ST-L
            S = torch.cat([Xtrain,Ytrain], dim = 0).to(device)
            model_c = fitC2ST(Xtrain, Ytrain, d_in,num_neurons_c2st,2, batch_size_c2st, learning_rate_c, epochs_c2st, device)           
            # Train MMD - O
            sigma0 = train_MMD_O(S, N_total, learning_rate_MMD_O, N_epoch)
            # Train Deep MMD 
            model_u, sigma, sigma0_u, ep = train_deep_MMD(S, N_total, d_in, d_latent, d_out, learning_rate_MMD_D, N_epoch, device, dtype)
            # Learn sigma for JSD directly from the gram-matrix
            sigma_jsd = bestSigmaDivergence(Xtrain,Ytrain,typeDivergence = 'JSD', learning_rate = learning_rate_jsd, is_cuda = is_cuda,n_epochs = 100)
            # learn the FF to approximate JSD for 4 different n_RFF
            model_jsd_rff0 = bestSigmaDivergenceRFF(Xtrain,Ytrain,typeDivergence = 'JSD', n_RFF = n_RFF[0], learning_rate = learning_rate_jsd_rff, is_cuda = True,n_epochs = 100)
            model_jsd_rff1 = bestSigmaDivergenceRFF(Xtrain,Ytrain,typeDivergence = 'JSD', n_RFF = n_RFF[1], learning_rate = learning_rate_jsd_rff, is_cuda = True,n_epochs = 100)
            model_jsd_rff2 = bestSigmaDivergenceRFF(Xtrain,Ytrain,typeDivergence = 'JSD', n_RFF = n_RFF[2], learning_rate = learning_rate_jsd_rff, is_cuda = True,n_epochs = 100)
            model_jsd_rff3 = bestSigmaDivergenceRFF(Xtrain,Ytrain,typeDivergence = 'JSD', n_RFF = n_RFF[3], learning_rate = learning_rate_jsd_rff, is_cuda = True,n_epochs = 100)

            # _ , sigma_jsd = optimizedDivergence(Xtrain,Ytrain,typeDivergence = 'JSD') # using secant algorithm
            # Learn the kernel for deep JSD with different n_RFF
            model0 = learningDeepRFF(Xtrain,Ytrain, typeKernel='product',n_RFF=n_RFF[0], is_cuda=is_cuda)
            model1 = learningDeepRFF(Xtrain,Ytrain, typeKernel='product',n_RFF=n_RFF[1], is_cuda=is_cuda) 
            model2 = learningDeepRFF(Xtrain,Ytrain, typeKernel='product',n_RFF=n_RFF[2], is_cuda=is_cuda) 
            model3 = learningDeepRFF(Xtrain,Ytrain, typeKernel='product',n_RFF=n_RFF[3], is_cuda=is_cuda)  

            ## TESTING THE LEARNED KERNELS AND MODEL ON TESTING SETS
            for testIdx in range(numTestSets):
                # Use testing set to get the estimate and then compare to the distribution of permutations
                Xtest, Ytest = sample_blobs(samples)

                # Compute thresholds and power test
                # MMD- O
                Stest = torch.cat([Xtest,Ytest], dim = 0).to(device)
                h_adaptive, _, _ = TST_MMD_adaptive_bandwidth(Stest, permTestSize, N_total, Stest, 0, sigma0, significance, device, dtype)
                mmd_results[repIdx,testIdx,sIdx] = h_adaptive
                # Deep MMD
                h_u, _, _ = TST_MMD_u(model_u(Stest), permTestSize, N_total, Stest, sigma, sigma0_u, significance, device,
                                                dtype, ep)
                deep_mmd_results[repIdx,testIdx,sIdx] = h_u
                
                # C2ST-S
                c2st_s_results[repIdx,testIdx,sIdx] = C2ST_perm_test(Xtest,Ytest, model_c, permTestSize, significance, 's', device)    
                
                # C2ST-L
                c2st_l_results[repIdx,testIdx,sIdx] = C2ST_perm_test(Xtest,Ytest, model_c, permTestSize, significance, 'l',device)

                # collect JSD baseline and permutation test
                jsd_ = QJSD(Xtest,Ytest, sigma = sigma_jsd)
                thr_jsd = jsd_permutation_test(Xtest, Ytest, sigma_jsd, significance, permTestSize)
                # Now we do it for the four models with Forier Features (we can use the same function of deep jsd because theres a network that computes the mapping explicitely)
                jsd_0 = deep_JSD(Xtest.to(device),Ytest.to(device),model_jsd_rff0)
                jsd_1 = deep_JSD(Xtest.to(device),Ytest.to(device),model_jsd_rff1)
                jsd_2 = deep_JSD(Xtest.to(device),Ytest.to(device),model_jsd_rff2)
                jsd_3 = deep_JSD(Xtest.to(device),Ytest.to(device),model_jsd_rff3)
                thr_jsd0 = deep_JSD_permutation_test(Xtest.to(device), Ytest.to(device), model_jsd_rff0, significance, permTestSize)
                thr_jsd1 = deep_JSD_permutation_test(Xtest.to(device), Ytest.to(device), model_jsd_rff1, significance, permTestSize)
                thr_jsd2 = deep_JSD_permutation_test(Xtest.to(device), Ytest.to(device), model_jsd_rff2, significance, permTestSize)
                thr_jsd3 = deep_JSD_permutation_test(Xtest.to(device), Ytest.to(device), model_jsd_rff3, significance, permTestSize)
                if thr_jsd < jsd_:
                    jsd_results[repIdx,testIdx,4,sIdx] = 1   ## indexing in the fourth position the original jsd to preserve the order
                if thr_jsd0 < jsd_0:
                    jsd_results[repIdx,testIdx,0,sIdx] = 1
                if thr_jsd1 < jsd_1:
                    jsd_results[repIdx,testIdx,1,sIdx] = 1
                if thr_jsd2 < jsd_2:
                    jsd_results[repIdx,testIdx,2,sIdx] = 1
                if thr_jsd3 < jsd_3:
                    jsd_results[repIdx,testIdx,3,sIdx] = 1

                # Now, for the deep JSD models              
                deep_jsd_0 = deep_JSD(Xtest.to(device),Ytest.to(device),model0)
                deep_jsd_1 = deep_JSD(Xtest.to(device),Ytest.to(device),model1)
                deep_jsd_2 = deep_JSD(Xtest.to(device),Ytest.to(device),model2)
                deep_jsd_3 = deep_JSD(Xtest.to(device),Ytest.to(device),model3)

                thr_deep_jsd0 = deep_JSD_permutation_test(Xtest.to(device), Ytest.to(device), model0, significance, permTestSize)
                thr_deep_jsd1 = deep_JSD_permutation_test(Xtest.to(device), Ytest.to(device), model1, significance, permTestSize)
                thr_deep_jsd2 = deep_JSD_permutation_test(Xtest.to(device), Ytest.to(device), model2, significance, permTestSize)
                thr_deep_jsd3 = deep_JSD_permutation_test(Xtest.to(device), Ytest.to(device), model3, significance, permTestSize)
                
                if thr_deep_jsd0 < deep_jsd_0:
                    deep_jsd_results[repIdx,testIdx,0,sIdx] = 1
                if thr_deep_jsd1 < deep_jsd_1:
                    deep_jsd_results[repIdx,testIdx,1,sIdx] = 1
                if thr_deep_jsd2 < deep_jsd_2:
                    deep_jsd_results[repIdx,testIdx,2,sIdx] = 1
                if thr_deep_jsd3 < deep_jsd_3:
                    deep_jsd_results[repIdx,testIdx,3,sIdx] = 1
                                

                        
    return jsd_results, deep_jsd_results, mmd_results, deep_mmd_results, c2st_s_results, c2st_l_results
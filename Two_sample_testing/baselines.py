## Baselines
# Code from https://github.com/fengliu90/DK-for-TST/blob/1c4065e81fb902e46e3316bfd98eadd0b7f22d74/utils.py#L55
import numpy as np
import torch
import torch.utils.data
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


is_cuda = True

class ModelLatentF(torch.nn.Module):
    """define deep networks."""
    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False

        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant

def get_item(x, is_cuda):
    """get the numpy value from a torch tensor."""
    if is_cuda:
        x = x.cpu().detach().numpy()
    else:
        x = x.detach().numpy()
    return x

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
    """compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None

    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4*(V1 - V2**2)
    if  varEst == 0.0:
        print('error!!'+str(V1))

    return mmd2, varEst, Kxyxy

def MMDu(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon=10 ** (-10), is_smooth=True, is_var_computed=True, use_1sample_U=True):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2
    L = 1 # generalized Gaussian (if L>1)

    nx = X.shape[0]
    ny = Y.shape[0]
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)

    if is_smooth:
        Kx = (1-epsilon) * torch.exp(-(Dxx / sigma0)**L -Dxx_org / sigma) + epsilon * torch.exp(-Dxx_org / sigma)
        Ky = (1-epsilon) * torch.exp(-(Dyy / sigma0)**L -Dyy_org / sigma) + epsilon * torch.exp(-Dyy_org / sigma)
        Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0)**L -Dxy_org / sigma) + epsilon * torch.exp(-Dxy_org / sigma)
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)

    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

def MMDu_linear_kernel(Fea, len_s, is_var_computed=True, use_1sample_U=True):
    """compute value of (deep) lineaer-kernel MMD and std of (deep) lineaer-kernel MMD using merged data."""
    try:
        X = Fea[0:len_s, :]
        Y = Fea[len_s:, :]
    except:
        X = Fea[0:len_s].unsqueeze(1)
        Y = Fea[len_s:].unsqueeze(1)
    Kx = X.mm(X.transpose(0,1))
    Ky = Y.mm(Y.transpose(0,1))
    Kxy = X.mm(Y.transpose(0,1))
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

def mmd2_permutations(K, n_X, permutations=500):
    """
        Fast implementation of permutations using kernel matrix.
    """
    K = torch.as_tensor(K)
    n = K.shape[0]
    assert K.shape[0] == K.shape[1]
    n_Y = n_X
    assert n == n_X + n_Y
    w_X = 1
    w_Y = -1
    ws = torch.full((permutations + 1, n), w_Y, dtype=K.dtype, device=K.device)
    ws[-1, :n_X] = w_X
    for i in range(permutations):
        ws[i, torch.randperm(n)[:n_X].numpy()] = w_X
    biased_ests = torch.einsum("pi,ij,pj->p", ws, K, ws)
    if True:  # u-stat estimator
        # need to subtract \sum_i k(X_i, X_i) + k(Y_i, Y_i) + 2 k(X_i, Y_i)
        # first two are just trace, but last is harder:
        is_X = ws > 0
        X_inds = is_X.nonzero()[:, 1].view(permutations + 1, n_X)
        Y_inds = (~is_X).nonzero()[:, 1].view(permutations + 1, n_Y)
        del is_X, ws
        cross_terms = K.take(Y_inds * n + X_inds).sum(1)
        del X_inds, Y_inds
        ests = (biased_ests - K.trace() + 2 * cross_terms) / (n_X * (n_X - 1))
    est = ests[-1]
    rest = ests[:-1]
    p_val = (rest > est).float().mean()
    return est.item(), p_val.item(), rest

def TST_MMD_adaptive_bandwidth(Fea, N_per, N1, Fea_org, sigma, sigma0, alpha, device, dtype):
    """run two-sample test (TST) using ordinary Gaussian kernel."""
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, is_smooth=False)
    mmd_value = get_item(TEMP[0],is_cuda)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        # print(Kx)
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]

        TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
        mmd_vector[r] = TEMP[0]
        if mmd_vector[r] > mmd_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(mmd_vector)
        #        print(np.int(np.ceil(N_per*alpha)))
        threshold = 0 # # ignorign this lines that sometimes causes issues --> S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, mmd_value.item()

def TST_MMD_u(Fea, N_per, N1, Fea_org, sigma, sigma0, alpha, device, dtype, epsilon=10 ** (-10),is_smooth=True):
    """run two-sample test (TST) using deep kernel kernel."""
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, epsilon,is_smooth)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    mmd_value_nn, p_val, rest = mmd2_permutations(Kxyxy, nx, permutations=N_per)
    if p_val > alpha:
        h = 0
    else:
        h = 1
    threshold = "NaN"
    return h,threshold,mmd_value_nn

def TST_MMD_u_HD(Fea, N_per, N1, Fea_org, sigma, sigma0, ep, alpha, device, dtype, is_smooth=True):
    """run two-sample test (TST) using deep kernel kernel."""
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, ep, is_smooth)
    mmd_value = get_item(TEMP[0], is_cuda)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        # print(Kx)
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]

        TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
        mmd_vector[r] = TEMP[0]
        if mmd_vector[r] > mmd_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(mmd_vector)
        #        print(np.int(np.ceil(N_per*alpha)))
    threshold = 0 # to avoid problems -- originally was S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, mmd_value.item()

def TST_MMD_u_linear_kernel(Fea, N_per, N1, alpha, device, dtype):
    """run two-sample test (TST) using (deep) lineaer kernel kernel."""
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu_linear_kernel(Fea, N1)
    mmd_value = get_item(TEMP[0], is_cuda)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        # print(Kx)
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]
        TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
        mmd_vector[r] = TEMP[0]
        if mmd_vector[r] > mmd_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(mmd_vector)
        threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, mmd_value.item()

def C2ST_NN_fit(S,y,N1,x_in,H,x_out,learning_rate_C2ST,N_epoch,batch_size,device,dtype):
    """Train a deep network for C2STs."""
    N = S.shape[0]
    if is_cuda:
        model_C2ST = ModelLatentF(x_in, H, x_out).cuda()
    else:
        model_C2ST = ModelLatentF(x_in, H, x_out)
    w_C2ST = torch.randn([x_out, 2]).to(device, dtype)
    b_C2ST = torch.randn([1, 2]).to(device, dtype)
    w_C2ST.requires_grad = True
    b_C2ST.requires_grad = True
    optimizer_C2ST = torch.optim.Adam(list(model_C2ST.parameters()) + [w_C2ST] + [b_C2ST], lr=learning_rate_C2ST)
    criterion = torch.nn.CrossEntropyLoss()
    f = torch.nn.Softmax(dim = 1)
    ind = np.random.choice(N, N, replace=False)
    tr_ind = ind[:np.int(np.ceil(N * 1))]
    te_ind = tr_ind
    dataset = torch.utils.data.TensorDataset(S[tr_ind,:], y[tr_ind])
    dataloader_C2ST = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    len_dataloader = len(dataloader_C2ST)
    for epoch in range(N_epoch):
        data_iter = iter(dataloader_C2ST)
        tt = 0
        while tt < len_dataloader:
            # training model using source data
            data_source = data_iter.__next__()
            S_b, y_b = data_source
            output_b = model_C2ST(S_b).mm(w_C2ST) + b_C2ST
            loss_C2ST = criterion(output_b, y_b)
            optimizer_C2ST.zero_grad()
            loss_C2ST.backward(retain_graph=True)
            # Update sigma0 using gradient descent
            optimizer_C2ST.step()
            tt = tt + 1
        # if epoch % 100 == 0:
        #     print(criterion(model_C2ST(S).mm(w_C2ST) + b_C2ST, y).item())
    output = f((model_C2ST(S[te_ind,:]).mm(w_C2ST) + b_C2ST))
    pred = output.max(1, keepdim=True)[1]
    STAT_C2ST = abs(pred[:N1].type(torch.FloatTensor).mean() - pred[N1:].type(torch.FloatTensor).mean())
    return pred, STAT_C2ST, model_C2ST, w_C2ST, b_C2ST

def TST_C2ST(S,N1,N_per,alpha,model_C2ST, w_C2ST, b_C2ST,device,dtype):
    """run C2ST-S."""
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    N = S.shape[0]
    f = torch.nn.Softmax(dim = 1)
    output = f(model_C2ST(S).mm(w_C2ST) + b_C2ST)
    pred_C2ST = output.max(1, keepdim=True)[1]
    STAT = abs(pred_C2ST[:N1].type(torch.FloatTensor).mean() - pred_C2ST[N1:].type(torch.FloatTensor).mean())
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        # print(indx)
        STAT_vector[r] = abs(pred_C2ST[ind_X].type(torch.FloatTensor).mean() - pred_C2ST[ind_Y].type(torch.FloatTensor).mean())
    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT

def TST_LCE(S,N1,N_per,alpha,model_C2ST, w_C2ST, b_C2ST,device,dtype):
    """run C2ST-L."""
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    N = S.shape[0]
    f = torch.nn.Softmax(dim = 1)
    output = f(model_C2ST(S).mm(w_C2ST) + b_C2ST)
    STAT = abs(output[:N1,0].type(torch.FloatTensor).mean() - output[N1:,0].type(torch.FloatTensor).mean())
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        # print(indx)
        STAT_vector[r] = abs(output[ind_X,0].type(torch.FloatTensor).mean() - output[ind_Y,0].type(torch.FloatTensor).mean())
    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT



def train_MMD_O(S, N1, learning_rate_MMD_O, N_epoch,initial_sigma = 'median'):
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        d = S.shape[1]
        if initial_sigma == 'median':
            Dxy = Pdist2(S[:N1,:],S[N1:,:])
            sigma0 = Dxy.median() * (2**(-2.1))
            sigma0 = sigma0.to(S.device)
        if initial_sigma == 'dimension':
            sigma0 = 2*d * torch.rand([1])
            sigma0 = sigma0.to(S.device)

        sigma0.requires_grad = True
        optimizer_sigma0 = torch.optim.Adam([sigma0], lr=learning_rate_MMD_O)
        for t in range(N_epoch):
            TEMPa = MMDu(S, N1, S, 0, sigma0, is_smooth=False)
            mmd_value_tempa = -1 * (TEMPa[0]+10**(-8))
            mmd_std_tempa = torch.sqrt(TEMPa[1]+10**(-8))
            STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
            optimizer_sigma0.zero_grad()
            STAT_adaptive.backward(retain_graph=True)
            optimizer_sigma0.step()

        return sigma0
def train_deep_MMD(S, N1, d_in, d_latent, d_out, learning_rate, N_epoch, device, dtype,initial_sigma = 'random', validation = False, Xval = None, Yval = None):
    # Train D+C
    np.random.seed(1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)

    # Initialize parameters
    if is_cuda:
        model_u = ModelLatentF(d_in, d_latent, d_out).cuda()
    else:
        model_u = ModelLatentF(d_in, d_latent, d_out)

    if initial_sigma == 'dimension':
        epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * d_in), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.1), device, dtype)
        sigma0OPT.requires_grad = True
    if initial_sigma == 'higgs':
        epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2*d_in), device, dtype)  # d = 3,5 ??
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
        sigma0OPT.requires_grad = False
    elif initial_sigma == 'random':
    #In their paper they use this initial parameters for the blobs dataset    
        epsilonOPT = MatConvert(np.random.rand(1) * (10 ** (-10)), device, dtype)
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.sqrt(np.random.rand(1) * 0.3), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.sqrt(np.random.rand(1) * 0.002), device, dtype)
        sigma0OPT.requires_grad = True

    optimizer_u = torch.optim.Adam(list(model_u.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT], lr=learning_rate) #
    if validation:
        Xval = Xval
        Yval = Yval
        Sval = torch.cat([Xval,Yval], dim = 0).to(S.device())
        log_dir = "TST/MMD/n_" +str(numSamples) + "lr_"+str(learning_rate) +  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir)
    
    
    # Train deep kernel to maximize test power
    for t in range(N_epoch):
        # Compute epsilon, sigma and sigma_0
        ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
        sigma = sigmaOPT ** 2
        sigma0_u = sigma0OPT ** 2
        # Compute output of the deep network
        modelu_output = model_u(S)
        # Compute J (STAT_u)
        TEMP = MMDu(modelu_output, N1, S, sigma, sigma0_u, ep)
        mmd_value_temp = -1 * (TEMP[0]+10**(-8))# -1 * TEMP[0]
        mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
        # Initialize optimizer and Compute gradient
        optimizer_u.zero_grad()
        STAT_u.backward(retain_graph=True)
        # Update weights using gradient descent
        optimizer_u.step()
        if validation:
            STAT_u_val, STATU_u_perm = validate_MMD(modelu,  N1, Sval, sigma, sigma0_u, ep)
            writer.add_scalars('MMD', {'training':-1* STAT_u, 'validation': -1*STAT_u_val, 'permutations': -1*STATU_u_perm}, t)
            writer.add_scalar('eps', ep.item(), t)

    return model_u, sigma.item(), sigma0_u.item(), ep.item()

def validate_MMD(modelu_output, N1, Sval, sigma, sigma0_u, ep):
    modelu_output = model_u(S)
    TEMP = MMDu(modelu_output, N1, Sval, sigma, sigma0_u, ep)
    mmd_value_temp = -1 * (TEMP[0]+10**(-8))# -1 * TEMP[0]
    mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
    STAT_u_val = torch.div(mmd_value_temp, mmd_std_temp)
    # permutations
    idxPerm = torch.randperm(len(Sval))
    Sperm = Sval[idxPerm,:]

    modelu_output = model_u(Sperm)
    TEMP = MMDu(modelu_output, N1, Sperm, sigma, sigma0_u, ep)
    mmd_value_temp = -1 * (TEMP[0]+10**(-8))# -1 * TEMP[0]
    mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
    STAT_u_perm = torch.div(mmd_value_temp, mmd_std_temp)
    return STAT_u_val, STATU_u_perm





## C2ST baseline
class classifierC2ST(nn.Module):
    def __init__(self,x_in, H, x_out):
        super( classifierC2ST, self).__init__()
        num_classes = 2
        
        self.fc1 = nn.Linear(x_in, H ) 
        self.fc2 = nn.Linear(H, H) 
        self.fc3 = nn.Linear(H, num_classes) 

    def forward(self, x):        
        x = F.relu(self.fc1(x)) #relu activation function
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
## C2ST baseline
class classifierC2ST2(nn.Module):
    def __init__(self,x_in, H, x_out):
        super( classifierC2ST2, self).__init__()
        num_classes = 2
        
        self.fc1 = nn.Linear(x_in, H ) 
        self.fc2 = nn.Linear(H, H) 
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, num_classes)  

    def forward(self, x):        
        x = F.relu(self.fc1(x)) #relu activation function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x 

# train and test for classification network
def train_loop(dataloader, model, optimizer,device):
    num_classes= 2 
    size = len(dataloader.dataset)
    train_loss, correct = 0, 0 
    for batch, (train_features, train_labels) in enumerate(dataloader):
        # forward pass
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)
        outputs = model(train_features).reshape([-1,num_classes]) 
        pred = F.log_softmax(outputs, dim=1)
        this_batch_size = len(train_labels)
        loss = F.nll_loss( pred, 
                  train_labels.reshape([this_batch_size]), reduction='sum')
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward() #compute both the loss and the gradient fields of trainable parameters
        optimizer.step() #apply the stochastic optimization scheme
        
        train_loss += loss.item()    
        correct += (pred.argmax(1) == train_labels).type(torch.float).sum().item()
    
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(train_features)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= size
    correct /= size  
    #print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")  
    return train_loss, correct

def test_loop(dataloader, model,device):
    num_classes= 2 
    size = len(dataloader.dataset)
    #batch_size = dataloader.batch_size
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (test_features, test_labels) in dataloader:
            # forward pass
            test_features = test_features.to(device)
            test_labels = test_labels.to(device)
            outputs = model(test_features).reshape([-1,num_classes])
            pred = F.log_softmax(outputs, dim=1)
            
            this_batch_size = len(test_labels)
            loss = F.nll_loss( pred, 
                  test_labels.reshape([this_batch_size]), reduction='sum')
        
            test_loss += loss.item()
            correct += (pred.argmax(1) == test_labels).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  
    return test_loss, correct


# the bootstrap procedure to estimate test threshold (permutation test)
def perm_test(gte, labels_te, numperm,significance):
    ntr = gte.shape[0]
    nte = ntr
    nXte = int(ntr/2)
    idx1_te = (labels_te == 0).nonzero(as_tuple=True)[0].cpu().numpy()
    idx2_te = (labels_te == 1).nonzero(as_tuple=True)[0].cpu().numpy()
    eta = np.mean(gte[idx1_te])-np.mean(gte[idx2_te])
        
    # test-only boostrap (permutation test)
    etastore = np.float32( np.zeros(numperm) )
    for iboot in range(numperm):
        tmp=torch.randperm(ntr).numpy()
        idx1_perm, idx2_perm = tmp[0:nXte], tmp[nXte:nte] 
        eta_perm = np.mean(gte[idx1_perm])-np.mean(gte[idx2_perm])
        etastore[iboot] = eta_perm
        
    talpha = np.quantile(etastore, (1-significance)) #test level alpha=0.05
    return eta, etastore, talpha

def fitC2ST(Xtr, Ytr, x_in,H,x_out, batch_size, learning_rate_c, epochs, device):
    ### C2ST-S test: training ###
    nXtr = Xtr.shape[0]
    nYtr = Ytr.shape[0]
    data_tr = torch.tensor( np.concatenate( (Xtr,  Ytr), axis=0) ).to(device)
    labels_tr = torch.tensor(np.concatenate( (np.zeros(nXtr),  np.ones(nYtr)), axis=0), dtype=int).to(device)   
    model_c = classifierC2ST2(x_in, H, x_out).to(device) #init model
    # model_c = ModelLatentF(x_in, H, x_out).to(device)
    training_data = torch.utils.data.TensorDataset(data_tr, labels_tr )
    train_dataloader = torch.utils.data.DataLoader(training_data, 
                                    batch_size= batch_size, shuffle=True)
    
    #init model
    model_c = classifierC2ST2(x_in, H, x_out).to(device) 
    # model_c = ModelLatentF(x_in, H, x_out).to(device) 
    # optimizer
    optimizer_c = optim.Adam(model_c.parameters(), 
                                lr= learning_rate_c)
    model_c.train()

    # loop of training
    train_loss_all, train_acc_all  = np.zeros(epochs), np.zeros(epochs)
    for t in range(epochs):
        #print(f"Epoch {t+1}\n-------------------------------")
        train_loss_all[t], train_acc_all[t]=train_loop(train_dataloader, model_c, optimizer_c,device)
    return model_c

def C2ST_perm_test(Xte,Yte, model_c, m_perm, significance, variant,device):
    nXte = Xte.shape[0]
    nYte = Yte.shape[0]
    nte = nXte + nYte
    data_te =  torch.tensor(np.concatenate( (Xte,  Yte), axis=0)).to(device)
    labels_te = torch.tensor(np.concatenate( (np.zeros(nXte),  np.ones(nYte)), axis=0), dtype=int).to(device)
    # test statistic
    with torch.no_grad():
        num_classes = 2
        # forward pass
        outputs_te = model_c(data_te).reshape([-1,num_classes])
        pred_te = F.log_softmax(outputs_te, dim=1)
        correct = (pred_te.argmax(1) == labels_te).type(torch.float).sum().item()
        test_acc =correct/nte 
    
    logit_te = outputs_te.detach().cpu().numpy()

    if variant == 's':
        ### C2ST-S test
        gte = np.float32(np.sign(logit_te[:,0])>0)
        eta, etastore, talpha = perm_test(gte, labels_te, m_perm, significance );
        # record the voet by rejection rule
        H =  np.float32( eta > talpha );
    if variant == 'l':
        ### C2ST-L test ###
        gte = logit_te[:,0]-logit_te[:,1];
        eta, etastore, talpha = perm_test(gte, labels_te, m_perm, significance );
        H =  np.float32( eta > talpha );
    return H 
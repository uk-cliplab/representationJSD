import numpy as np
import torch

class RFF_tuneable(torch.nn.Module):
    def __init__(self, d_latent, d_out,sigma_init,variant = 'cosine_sine',freezeRFF=False, seed = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        w_init = torch.randn([d_latent,d_out], dtype = torch.float )
        bias_init = 2*np.pi*torch.rand(d_out, dtype=torch.float)

        if freezeRFF:
            self.weight = torch.nn.Parameter(w_init,requires_grad = False)
            self.bias = torch.nn.Parameter(bias_init,requires_grad = False)
        else:
            self.weight = torch.nn.Parameter(w_init,requires_grad = True)
            self.bias = torch.nn.Parameter(bias_init,requires_grad = True)

        self.sigma = torch.nn.Parameter(sigma_init, requires_grad=False)
        self.d_out = d_out # necessary to normalize the FF
        self.variant = variant

    def forward(self, x):
        if self.variant == 'cosine':
            w_times_x= torch.matmul(x, (1/torch.sqrt(self.sigma))*self.weight) + self.bias
            return np.sqrt(2/self.d_out)*(torch.cos(w_times_x))
        elif self.variant == 'cosine_sine':
            #w_times_x= torch.matmul(x, (1/torch.sqrt(self.sigma.data))*self.weight.data)
            w_times_x= torch.matmul(x, (1/self.sigma)*self.weight)

            return np.sqrt(1/self.d_out)*torch.cat((torch.cos(w_times_x), torch.sin(w_times_x)), dim=-1)
        else:
            print('False variant, choose between cosine or cosine_sine') 
            return None

class DeepFourierFeatures(torch.nn.Module):
    """define deep networks."""
    def __init__(self, d_in, H, d_latent,n_RFF,sigma, typeKernel = 'product', variant = 'cosine_sine', freezeRFF = False):
        """Init latent features."""
        super().__init__()
        self.restored = False
        eps_ = torch.tensor(0.0) #0.5 after normalization (-2.0 seems to work well for HDGM)
        self.eps = torch.nn.Parameter(eps_,requires_grad = False)
        self.typeKernel = typeKernel

        self.latent = torch.nn.Sequential(
            torch.nn.Linear(d_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, d_latent, bias=True),
        )
        if typeKernel == 'product':
            self.RFF_latent_input = RFF_tuneable(d_latent + d_in,n_RFF, sigma, variant = variant,freezeRFF = freezeRFF)
        elif typeKernel == 'average':
            self.RFF_latent = RFF_tuneable(d_latent,n_RFF, sigma, variant = variant, freezeRFF=freezeRFF)
        else:
            print("got false type of Kernel: choose product or average")
        self.RFF_input = RFF_tuneable(d_in,n_RFF, sigma, variant = variant, freezeRFF = freezeRFF)


    def forward(self, input):
        """Forward the Net."""
        eps_exp = torch.exp(self.eps) / (1 + torch.exp(self.eps))
        phiLatent = self.latent(input)
        phiFFInput = self.RFF_input(input) # mapping from Fourier features to the input
        if self.typeKernel == 'product':
            # this represents the mapping of the product of the two kernels, (K(phiw(x),phiw(x'))*q(x,x'))
            # for gaussian kernels is the kernel of the concatenation of features
            combinedFeat = torch.cat((input,phiLatent), dim =-1) 
            phiProduct = self.RFF_latent_input(combinedFeat)
            # the sum of kernels is equivalent to the concatenation of the features in the feature space
            phi =  torch.cat((torch.sqrt(1-eps_exp)*phiProduct,torch.sqrt(eps_exp)*phiFFInput), dim = -1) 
                       
        elif self.typeKernel == 'average':
            phiFFLatent = self.RFF_latent(phiLatent)
            phi = torch.cat((torch.sqrt(1-eps_exp)*phiFFLatent,torch.sqrt(eps_exp)*phiFFInput), dim = -1)          
        return phi

class DeepFourierFeaturesNetwork(torch.nn.Module):
    """define deep networks."""
    def __init__(self, d_in, H, d_latent,n_RFF,sigma, variant = 'cosine_sine', freezeRFF = False):
        """Init latent features."""
        super().__init__()


        self.latent = torch.nn.Sequential(
            torch.nn.Linear(d_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, d_latent, bias=True),
        )

        self.RFF_latent = RFF_tuneable(d_latent,n_RFF, sigma, variant = variant, freezeRFF=freezeRFF)


    def forward(self, input):
        """Forward the Net."""

        phiLatent = self.latent(input)
        phi = self.RFF_latent(phiLatent)        
        return phi

class RFF_layer(torch.nn.Module):
    def __init__(self, d_in, d_out,sigma_init,variant = 'cosine_sine',freezeRFF=False, seed = None):
        super().__init__()
        self.RFF_input = RFF_tuneable(d_in, d_out, sigma_init, variant, freezeRFF, seed = seed)

    def forward(self, x):
        phi = self.RFF_input(x)
        return phi


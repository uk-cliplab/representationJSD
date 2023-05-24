import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

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




def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

def weights_init_DFFN(w):
    """
    Initializes the weights of the layer, w.
    """
    if isinstance(w, nn.Conv2d) or isinstance(w, nn.Linear):
        nn.init.normal_(w.weight.data, 0.0, 1.0)







class Generator(nn.Module):
    def __init__(self, imgSize, nz, ngf, nc):
        super().__init__()
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,    ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Sigmoid()
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, imgSize, ndf, nc):
        super().__init__()
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 1).squeeze(1)
        return output


# # Define the Discriminator Network
# class Discriminator_FF(nn.Module):
#     def __init__(self, params):
#         super().__init__()

#         # Input Dimension: (nc) x 64 x 64
#         self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
#             4, 2, 1, bias=False)

#         # Input Dimension: (ndf) x 32 x 32
#         self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
#             4, 2, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(params['ndf']*2)

#         # Input Dimension: (ndf*2) x 16 x 16
#         self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
#             4, 2, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(params['ndf']*4)

#         # Input Dimension: (ndf*4) x 8 x 8
#         self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
#             4, 2, 1, bias=False)
#         self.bn4 = nn.BatchNorm2d(params['ndf']*8)

#         # Input Dimension: (ndf*8) x 4 x 4
#         self.conv5 = nn.Conv2d(params['ndf']*8, 1, 4, 1, 0, bias=False)

#     def forward(self, x):
#         x = F.leaky_relu(self.conv1(x), 0.2, True)
#         x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
#         x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
#         x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
#         # flatten previous layer
#         x = x.view(x.size(0), -1)
#         # x = F.sigmoid(self.conv5(x))

#         return x

class Discriminator_FF(nn.Module):
    def __init__(self,  nc, ndf):
        super().__init__()
        
        self.main = nn.Sequential(

            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Tanh(),
            # nn.Dropout(0.2),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            # nn.Tanh(),
            
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.Tanh(),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Tanh(),
            # state size. ``(ndf*8) x 4 x 4``
            # nn.Conv2d(ndf * 8, 32, 4, 1, 0, bias=False),
            # nn.Dropout(0.2),
            # flatten previous layer to pass to the linear layer
            # # state size. (ndf*8) x 4 x 4


            nn.Flatten(),
            nn.Linear(ndf*8*4*4, 8*ndf),
            # nn.BatchNorm1d(8*ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Tanh(),
            nn.Linear(8*ndf, 4*ndf),
            
            # nn.Dropout(0.1),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.AdaptiveAvgPool2d(1),
            # nn.Flatten(),
            # nn.Linear(ndf*8, ndf),
            # nn.LeakyReLU(0.2, inplace=True),
            # # nn.Dropout(0.2),
            
        )
        
    def forward(self, input):
        output = self.main(input)
        output = output.view(input.size(0), -1)
        return output
    

class DeepFourierFeaturesNetwork_Conv(torch.nn.Module):
    """define deep networks."""
    def __init__(self, nc,  ndf, FF):
        """Init latent features."""
        super().__init__()
        self.latent = Discriminator_FF( nc, ndf)

        self.RFF_latent = RFF_tuneable(4*ndf,FF, torch.tensor(2.0), freezeRFF= False)

    def forward(self, img):
        """Forward the Net."""
        # img_flat = img.view(img.size(0), -1)
        phiLatent = self.latent(img)
        # print(phiLatent[:1,:1])
        phi = self.RFF_latent(phiLatent)        
        return phi
    

class DeepFourierFeatures_residual(torch.nn.Module):
    """define deep networks."""
    def __init__(self, nc,  ndf, FF):
        """Init latent features."""
        super().__init__()

        d_in = nc*64*64
        self.restored = False
        eps_ = torch.tensor(-2.0) #0.5 after normalization (-2.0 seems to work well for HDGM)
        self.eps = torch.nn.Parameter(eps_,requires_grad = True)

        self.latent =  Discriminator_FF( nc, ndf)

        self.RFF_latent_input = RFF_tuneable(ndf + d_in,FF, torch.tensor(np.sqrt(2*(ndf))),freezeRFF = False)

        self.RFF_input = RFF_tuneable( d_in,FF, torch.tensor(np.sqrt(2*d_in)), freezeRFF = False)


    def forward(self, input):
        """Forward the Net."""
        input_flat = input.view(input.size(0), -1)
        eps_exp = torch.exp(self.eps) / (1 + torch.exp(self.eps))
        phiLatent = self.latent(input)
        phiFFInput = self.RFF_input(input_flat) # mapping from Fourier features to the input

        # this represents the mapping of the product of the two kernels, (K(phiw(x),phiw(x'))*q(x,x'))
        # for gaussian kernels is the kernel of the concatenation of features
        combinedFeat = torch.cat((input_flat,phiLatent), dim =-1) 
        phiProduct = self.RFF_latent_input(combinedFeat)
        # the sum of kernels is equivalent to the concatenation of the features in the feature space
        phi =  torch.cat((torch.sqrt(1-eps_exp)*phiProduct,torch.sqrt(eps_exp)*phiFFInput), dim = -1)      
        return phi









### Fourier Feature network that work the best so far
class DeepFourierFeaturesNetwork_FC(torch.nn.Module):
    """define deep networks."""
    def __init__(self, img_shape, dout, FF):
        """Init latent features."""
        super().__init__()


        self.latent = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 2*dout),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2*dout, dout),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(256, 2), ## comment this line 
            # nn.Sigmoid(),
        )

        self.RFF_latent = RFF_tuneable(dout,FF, torch.tensor(10.0), freezeRFF= True) #256,100

    def forward(self, img):
        """Forward the Net."""
        img_flat = img.view(img.size(0), -1)
        phiLatent = self.latent(img_flat)
        # print(phiLatent[:3:3])
        phi = self.RFF_latent(phiLatent)        
        return phi




from argparse import ArgumentParser
import numpy as np
import pandas as pd
import seaborn as sns
colors = sns.color_palette()

## Code from https://github.com/ChristophReich1996/Mode_Collapse
# Manage command line arguments
parser = ArgumentParser()


parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], type=str,
                    help='Set device to be utilized. cuda or cpu.')
parser.add_argument('--epochs', default=500, type=int,
                    help='Training epochs to be performed.')
parser.add_argument('--d_updates', default=1, type=int,
                    help='Discriminator updates per generator update.')
parser.add_argument('--plot_frequency', default=10, type=int,
                    help='Frequency of epochs to produce plots.')
parser.add_argument('--lr_d', default=0.0001, type=float,
                    help='Discriminator learning rate to be applied.')
parser.add_argument('--lr_g', default=0.0001, type=float,
                    help='Generator learning rate to be applied.')
parser.add_argument('--latent_size', default=32, type=int,
                    help='Size of latent vector to be utilized.')
parser.add_argument('--samples', default=10000, type=int,
                    help='Number of samples from the real distribution.')
parser.add_argument('--batch_size', default=500, type=int,
                    help='Batch size to be utilized.')
parser.add_argument('--loss', default='standard', type=str,
                    choices=['standard', 'non-saturating', 'hinge', 'wasserstein', 'wasserstein-gp', 'least-squares', 'repJSD'],
                    help='GAN loss function to be used.')
parser.add_argument('--spectral_norm', default=False, action='store_true',
                    help='If set use spectral norm to stabilize discriminator.')
parser.add_argument('--clip_weights', default=0., type=float,
                    help='If > 0., weights will be clipped to [-clip_weights, clip_weights].')
parser.add_argument('--topk', default=False, action='store_true',
                    help='If set top-k training is utilized after 0.5 of the epochs to be performed.')

parser.add_argument('--noise_type', default='gaussian', type= str, choices = ['gaussian', 'uniform'], 
                    help='type of noise of the latent space')
parser.add_argument('--plot_loss', default=False, action='store_true', help='If set, plot loss')

# Get arguments
args = parser.parse_args()

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import utils
import loss

if __name__ == '__main__':
    # Make directory to save plots
    path = os.path.join(os.getcwd(), 'plots', args.loss + ("_top_k" if args.topk else "") + ("_sn" if args.spectral_norm else "") + ("_clip" if args.clip_weights else ""))
    os.makedirs(path, exist_ok=True)
    # Init hyperparameters
    

    # Get data
    data: torch.Tensor = utils.get_data(samples=args.samples).to(args.device)
    # Get generator
    generator: nn.Module = utils.get_generator(latent_size=args.latent_size)
    # Get discriminator
    if args.loss != 'repJSD':
        discriminator: nn.Module = utils.get_discriminator(use_spectral_norm=args.spectral_norm)
    else:
        discriminator: nn.Module = utils.get_DFFN(use_spectral_norm=args.spectral_norm)
    # Init Loss function
    if args.loss == 'standard':
        loss_generator: nn.Module = loss.GANLossGenerator()
        loss_discriminator: nn.Module = loss.GANLossDiscriminator()
    elif args.loss == 'non-saturating':
        loss_generator: nn.Module = loss.NSGANLossGenerator()
        loss_discriminator: nn.Module = loss.NSGANLossDiscriminator()
    elif args.loss == 'hinge':
        loss_generator: nn.Module = loss.HingeGANLossGenerator()
        loss_discriminator: nn.Module = loss.HingeGANLossDiscriminator()
    elif args.loss == 'wasserstein':
        loss_generator: nn.Module = loss.WassersteinGANLossGenerator()
        loss_discriminator: nn.Module = loss.WassersteinGANLossDiscriminator()
    elif args.loss == 'wasserstein-gp':
        loss_generator: nn.Module = loss.WassersteinGANLossGPGenerator()
        loss_discriminator: nn.Module = loss.WassersteinGANLossGPDiscriminator()
    elif args.loss == 'repJSD':
        loss_discriminator: nn.Module = loss.repJSDGANLossDiscriminator()
        loss_generator: nn.Module = loss.repJSDGANLossGenerator()
    else:
        loss_generator: nn.Module = loss.LSGANLossGenerator()
        loss_discriminator: nn.Module = loss.LSGANLossDiscriminator()
    # Networks to train mode
    generator.train()
    discriminator.train()
    # Models to device
    generator.to(args.device)
    discriminator.to(args.device)
    # Init optimizer
    generator_optimizer: torch.optim.Optimizer = torch.optim.RMSprop(generator.parameters(), lr=args.lr_d)    
    discriminator_optimizer: torch.optim.Optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr_g)
    # generator_optimizer: torch.optim.Optimizer = torch.optim.Adam(generator.parameters(), lr=2*args.lr, betas = (0.5,0.999))
    # discriminator_optimizer: torch.optim.Optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas = (0.5,0.999))
    # Init progress bar
    progress_bar = tqdm(total=args.epochs)
    # Training loop
    if args.noise_type == "gaussian":
        fixed_generator_noise: torch.Tensor = torch.randn([args.samples // 10, args.latent_size], device=args.device)
    else: 
        fixed_generator_noise: torch.Tensor = torch.rand([args.samples // 10, args.latent_size], device=args.device)
    step_n = 0

    loss_gen = np.zeros((args.epochs +1) * args.samples // args.batch_size)
    for epoch in range(args.epochs):  # type: int
        # Update progress bar
        progress_bar.update(n=1)
        # Update discriminator more often than generator to train it till optimality and get more reliable gradients of Wasserstein
        for index in range(0, args.samples, args.batch_size):  # type:int
        

            for _ in range(args.d_updates):  # type: int
                # Shuffle data
                data = data[torch.randperm(data.shape[0], device=args.device)]
            
                # Get batch
                batch: torch.Tensor = data[index:index + args.batch_size]
                # Get noise for generator
                if args.noise_type == "gaussian":
                    noise: torch.Tensor = torch.randn([args.batch_size, args.latent_size], device=args.device)
                else:
                    noise: torch.Tensor = torch.rand([args.batch_size, args.latent_size], device=args.device)
                # Optimize discriminator
                discriminator_optimizer.zero_grad()
                generator_optimizer.zero_grad()
                with torch.no_grad():
                    fake_samples: torch.Tensor = generator(noise)
                prediction_real: torch.Tensor = discriminator(batch)
                prediction_fake: torch.Tensor = discriminator(fake_samples)
                if isinstance(loss_discriminator, loss.WassersteinGANLossGPDiscriminator):
                    loss_d: torch.Tensor = loss_discriminator(prediction_real, prediction_fake, discriminator, batch,
                                                            fake_samples)
                else:
                    loss_d: torch.Tensor = loss_discriminator(prediction_real, prediction_fake)
                loss_d.backward()
                discriminator_optimizer.step()

                # Clip weights to enforce Lipschitz constraint as proposed in Wasserstein GAN paper
                if args.clip_weights > 0:
                    with torch.no_grad():
                        for param in [param for name, param in discriminator.named_parameters() if '8' not in name]:
                            param.clamp_(-args.clip_weights, args.clip_weights)

            # Get noise for generator
            if args.noise_type == "gaussian":
                noise: torch.Tensor = torch.randn([args.batch_size, args.latent_size], device=args.device)
            else:
                noise: torch.Tensor = torch.rand([args.batch_size, args.latent_size], device=args.device)
            # Optimize generator
            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            fake_samples: torch.Tensor = generator(noise)
            prediction_fake: torch.Tensor = discriminator(fake_samples)
            if args.topk and (epoch >= 0.5 * args.epochs):
                prediction_fake = torch.topk(input=prediction_fake[:, 0], k=prediction_fake.shape[0] // 2)[0]
            
            if isinstance(loss_generator, loss.repJSDGANLossGenerator):
                prediction_real: torch.Tensor = discriminator(batch)
                loss_g: torch.Tensor = loss_generator(prediction_real, prediction_fake)
            else:
                loss_g: torch.Tensor = loss_generator(prediction_fake)
            loss_g.backward()
            generator_optimizer.step()
            # Update progress bar description
            progress_bar.set_description(
                'Epoch {}, Generator loss {:.4f}, Discriminator loss {:.4f}'.format(epoch, loss_g.item(),
                                                                                   loss_d.item()))
            loss_gen [step_n] = loss_g.item()
            step_n = step_n +1
            

        # Plot samples of generator
        if ((epoch + 1) % args.plot_frequency) == 0 or step_n == 10000 :
            generator.eval()
            generator_samples = generator(fixed_generator_noise)
            generator_samples = generator_samples.cpu().detach().numpy()
            plt.rc('text', usetex=True)
            font = {'family' : 'serif',
                    'size'   : 18}
            plt.rc('font', **font)
            plt.scatter(data[::10, 0].cpu(), data[::10, 1].cpu(), color='blue', label='Real', s=2, alpha=0.5)
            plt.scatter(generator_samples[:, 0], generator_samples[:, 1], color='red',
                        label='Generated', s=2, alpha=0.5)
            # Remove gridlines
            plt.grid(visible = False)
            plt.legend(loc=1, markerscale=12.)

            plt.title('Standard',fontsize = 35)   # 'Step {}'.format(step_n) (epoch + 1) * args.samples // args.batch_size)
            plt.xlim((-1.5, 1.5))
            plt.ylim((-1.5, 1.5))
            #remove all axis and ticks
            plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
            # plt.grid()
            plt.savefig(os.path.join(path, '{}.pdf'.format(str(epoch + 1).zfill(3))), dpi=300, bbox_inches='tight')
            plt.close()
            generator.train()
        
    # Plot loss_gen and save the figure in the path folder
    if args.plot_loss:
        EMA_SPAN = 50
        loss_smooth = pd.Series(loss_gen[:step_n]).ewm(span=EMA_SPAN).mean()
        
        plt.plot(loss_gen[:step_n],  alpha=0.4)
        plt.plot(loss_smooth[:step_n], color=colors[0], linewidth=2)
        plt.xlabel('Step')
        plt.ylabel(r'$D_{JS}$')
        plt.title('Loss generator', fontsize = 35)
        plt.savefig(os.path.join(path, 'loss_gen.pdf'),dpi=300, bbox_inches='tight')
        plt.close()
    
    



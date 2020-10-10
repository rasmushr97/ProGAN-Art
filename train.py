from models import Discriminator, Generator
import torch
import torchvision.utils as utils
from tqdm import tqdm
import wandb

class Trainer:
    def __init__(self, zdim, features, channels, device, lr=0.001, betas=(0.0, 0.99), eps=1e-8, log=False):
        self.device = torch.device(device)
        self.G = Generator(zdim, features, channels, lr, betas, eps, device=self.device)
        self.D = Discriminator(features, channels, lr, betas, eps, device=self.device)

        self.zdim = zdim
        self.fixed_noise = torch.randn(32, zdim, 1, 1).to(self.device)

        self.epoch = 0

        self.log = log
        if self.log:
            wandb.init(project="art-gan")
            wandb.watch(self.G)
            wandb.watch(self.D)

    def train_epoch(self, dataloader):
        with tqdm(total=len(dataloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as pbar:
            for batch_idx, images in enumerate(dataloader):
                lossD, lossG = self._train_batch(images)

                pbar.set_description(f'Epoch: {self.epoch}, Dloss: {lossD:.3f}, Gloss: {lossG:.3f}')
                pbar.update(1)

                if self.log and batch_idx % 50 == 0:
                    self._log_generator()

        self.epoch += 1

    def _train_batch(self, reals):
        reals = reals.to(self.device)
        batch_size = reals.size(0)
        noise = torch.randn(batch_size, self.zdim, 1, 1).to(self.device)

        self.D.zero_grad()

        fakes = self.G(noise)

        output_real = self.D(reals).reshape(-1)
        output_fake = self.D(fakes.detach()).reshape(-1)

        lossD_real = -output_real.mean()
        lossD_fake = output_fake.mean()
        
        grad_penalty = self._gp_loss(reals, fakes)

        lossD = lossD_real + lossD_fake + grad_penalty

        lossD.backward(retain_graph=True)

        self.D.optim.step()

        self.G.zero_grad()

        output_fake = self.D(fakes).reshape(-1)

        lossG = -output_fake.mean()
        lossG.backward()
        self.G.optim.step()

        return lossD, lossG

    def _gp_loss(self, x, xf):
        N, C, H, W = x.size()
        eps = torch.rand(N, 1, 1, 1)
        eps = eps.expand(-1, C, H, W).to(self.device)
        interpolates = eps*x + (1-eps)*xf
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        yi = self.D(interpolates)
        yi = yi.sum()

        yi_grad = torch.autograd.grad(outputs=yi, inputs=interpolates,
                                        create_graph=True, retain_graph=True)

        yi_grad = yi_grad[0].view(N, -1)
        yi_grad = torch.norm(yi_grad, p=2, dim=1)
        gp = torch.pow(yi_grad-1., 2).mean()
        return gp * 10.0

    def _log_generator(self):
        fake = self.G(self.fixed_noise).detach().cpu()
        grid = utils.make_grid(fake, padding=2, normalize=True)
        wandb.log({'most_recent' : [wandb.Image(grid, caption="Generated")]})

    def set_alpha(self, alpha):
        self.D.set_alpha(alpha)
        self.G.set_alpha(alpha)

    def scale(self):
        self.D.add_scale()
        self.G.add_scale()

    def save(self, folder):
        torch.save(self.G, f'{folder}/generator.h5')
        torch.save(self.D, f'{folder}/discriminator.h5')
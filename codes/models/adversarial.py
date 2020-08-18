from types import SimpleNamespace

from models.discriminator import Discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import torch.optim as optim


def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

class Adversarial(nn.Module):
    def __init__(self, patch_size, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = 1
        self.dis = Discriminator(patch_size)
        if gan_type == 'WGAN_GP':
            # see https://arxiv.org/pdf/1704.00028.pdf pp.4
            optim_dict = {
                'optimizer': 'ADAM',
                'betas': (0, 0.9),
                'epsilon': 1e-8,
                'lr': 1e-5,
                'weight_decay': 0,
                'decay': 200,
                'gamma': 0.5
            }
            optim_args = SimpleNamespace(**optim_dict)
        else:
            optim_dict = {
                'optimizer': 'ADAM',
                'betas': (0.9, 0.99),
                'epsilon': 1e-8,
                'lr': 1e-5,
                'weight_decay': 0,
                'decay': '200-300',
                'gamma': 0.5
            }
            optim_args = SimpleNamespace(**optim_dict)

        self.optimizer = make_optimizer(optim_args, self.dis)

    def forward(self, fake, real):
        # updating discriminator...
        self.loss = 0
        fake_detach = fake.detach()     # do not backpropagate through G
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            # d: B x 1 tensor
            d_fake = self.dis(fake_detach)
            d_real = self.dis(real)
            retain_graph = False
            if self.gan_type == 'GAN':
                loss_d = self.bce(d_real, d_fake)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.dis(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty
            # from ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
            elif self.gan_type == 'RGAN':
                better_real = d_real - d_fake.mean(dim=0, keepdim=True)
                better_fake = d_fake - d_real.mean(dim=0, keepdim=True)
                loss_d = self.bce(better_real, better_fake)
                retain_graph = True

            # Discriminator update
            self.loss += loss_d.item()
            
            if loss_d>1e-2:
                loss_d.backward(retain_graph=retain_graph)
            self.optimizer.step()

            if self.gan_type == 'WGAN':
                for p in self.dis.parameters():
                    p.data.clamp_(-1, 1)

        self.loss /= self.gan_k

        # updating generator...
        d_fake_bp = self.dis(fake)      # for backpropagation, use fake as it is
        if self.gan_type == 'GAN':
            label_real = torch.ones_like(d_fake_bp)
            loss_g = F.binary_cross_entropy_with_logits(d_fake_bp, label_real)
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_bp.mean()
        elif self.gan_type == 'RGAN':
            better_real = d_real - d_fake_bp.mean(dim=0, keepdim=True)
            better_fake = d_fake_bp - d_real.mean(dim=0, keepdim=True)
            loss_g = self.bce(better_fake, better_real)

        # Generator loss
        return loss_g
    
    def state_dict(self, *args, **kwargs):
        state_discriminator = self.dis.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)

    def bce(self, real, fake):
        label_real = torch.ones_like(real)
        label_fake = torch.zeros_like(fake)
        bce_real = F.binary_cross_entropy_with_logits(real, label_real)
        bce_fake = F.binary_cross_entropy_with_logits(fake, label_fake)
        bce_loss = bce_real + bce_fake
        return bce_loss
               
# Some references
# https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# OR
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py

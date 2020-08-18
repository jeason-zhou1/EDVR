import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss,GANLoss
from models import vgg
from models.adversarial import Adversarial

logger = logging.getLogger('base')


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()
            self.loss = []
            losses = train_opt['pixel_criterion']
            for loss in losses.split('+'):
                weight, loss_type = loss.split('*')
                #### loss
                if loss_type == 'l1':
                    loss_function = nn.L1Loss().to(self.device)
                elif loss_type == 'l2':
                    loss_function = nn.MSELoss(reduction='sum').to(self.device)
                elif loss_type == 'cb':
                    loss_function = CharbonnierLoss().to(self.device)
                elif loss_type.find('VGG')>=0:
                    loss_function = vgg.VGG(loss_type[3:],255).to(self.device)
                elif loss_type.find('GAN')>=0:
                    loss_function = Adversarial(opt['datasets']['train']['GT_size'],loss_type).to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
                # self.l_pix_w = train_opt['pixel_weight']
                self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
                )
                if loss_type.find('GAN') >= 0:
                    self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})


            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        # print(step)
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)
        losses = []
        # log=''
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](self.fake_H, self.real_H)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                # self.optimizer_G.zero_grad()
                # effective_loss.backward()
                # self.optimizer_G.step()
                self.log_dict[l['type']] = effective_loss.item()
                # log += l['type']+':'+str(loss.item())+';'
                # self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                # self.log[-1, i] += self.loss[i - 1]['function'].loss# adversarial的鉴别损失
                self.log_dict[l['type']] = self.loss[i - 1]['function'].loss
        # print('the length of the losses:',len(losses))
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log_dict['total'] = loss_sum.item()
        loss_sum.backward()
        self.optimizer_G.step()


        # set log

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

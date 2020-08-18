import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch.nn.init as init

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def de_pixelshuffle(input, downscale_factor):     # channal
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, downscale_factor, out_width, downscale_factor)
    channels *= downscale_factor ** 2
    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_height, out_width)
    return shuffle_out


class PFBlock(nn.Module):
    def __init__(self,nf,n_frame):
        super(PFBlock,self).__init__()
        # self.conv0 = nn.Conv2d(nf,nf,5,padding=2)
        self.nf = nf*n_frame
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.n_frame = n_frame
        self.conv1 = nn.Conv2d(n_frame*nf,n_frame*nf,3,padding=1,groups=n_frame)
        self.conv10 = nn.Conv2d(nf*n_frame,nf,1)
        self.conv11 = nn.Conv2d(nf*2*n_frame,n_frame*nf,3,padding=1,groups=n_frame)
        initialize_weights([self.conv1,self.conv10,self.conv11],0.1)

    def forward(self,x):
        # print(x.shape)# N,nf*nframe,h,w
        base = x
        x = self.lrelu(self.conv1(x))
        temp = x
        # print(x.shape)# N,nf*nframe,h,w
        x = self.lrelu(self.conv10(x))
        
        # print(x.shape)# N,NF,h,w
        x_each = torch.split(temp,self.nf//self.n_frame,dim=1)
        x_cat = [torch.cat([each,x],dim=1) for each in x_each]
        # print(len(x_each),x_each[0].shape)
        x_cat = torch.cat(x_cat,dim=1)
        
        out = self.conv11(x_cat)
        # print(out.shape)
        return out+base
        
class NONLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian',
                 sub_sample=False, bn_layer=False):
        super(NONLocalBlock2D, self).__init__()
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        self.mode = mode

        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool = nn.MaxPool2d
        sub_sample = nn.Upsample
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                            kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)

        self.operation_function = self._embedded_gaussian

        self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
        self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):

        output = self.operation_function(x)
        return output
        
    def _embedded_gaussian(self, x):
        batch_size,C,H,W = x.shape

        # g(x)同样把通道数减为了一半
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        # (b,hw,0.5c)
        g_x = g_x.permute(0, 2, 1)

        # 2D卷积 theta，此处的dimension是2，将通道数变成了原来的一半,(b,0.5c,hw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        # 此处进行维度变换是为了方便进行矩阵相乘,(b,hw,0.5c)
        theta_x = theta_x.permute(0, 2, 1)
        # phi的操作也是将通道数变成原来的一半，phi和theta的操作是一样的,(b,0.5c,hw)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # print(phi_x.shape,theta_x.shape)
        f = torch.matmul(theta_x, phi_x)
        # return f，
        # f_div_C相当于是一个在space上的一个的权重，（b,hw,hw）
        f_div_C = F.softmax(f, dim=-1)
        # return f_div_C
        # (b, hw,hw)dot(b,hw,0.5c) ==> (b,hw,0.5c)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        # (b,0.5c,h,w)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        # W，将通道变回原来两倍 (b,0.5c,h,w)==> (b,c,h,w)
        W_y = self.W(y)
        z = W_y + x

        return z

class Nonlocal(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, reduction=8,sub_sample=False, bn_layer=True):
        super(Nonlocal, self).__init__()
        self.non_local = (NONLocalBlock2D(in_channels=in_feat,inter_channels=inter_feat, sub_sample=sub_sample,bn_layer=bn_layer))

        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        ## divide feature map into 4 part
        batch_size,C,H,W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]

        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return  nonlocal_feat

        
class PFNL(nn.Module):
    def __init__(self,nf=64, nframes=7, n_blocks=20):
        super(PFNL,self).__init__()
        self.scale = 4
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_first = nn.Conv2d(3*nframes,nf*nframes,5,padding=2,groups=nframes)
        self.non_local = NONLocalBlock2D(nframes*3*4)

        layers = []
        PFB = functools.partial(PFBlock,nf=nf,n_frame=nframes)
        for _ in range(n_blocks):
            layers.append(PFB())
        self.PFB = nn.Sequential(*layers)
        self.trunk = nn.Conv2d(nframes*nf,nf,3,padding=1)
        self.conv_upsample1 = nn.Conv2d(16,64,3,padding=1)
        self.conv_upsample2 = nn.Conv2d(16,64,3,padding=1)
        self.conv_last = nn.Conv2d(64,3,3,padding=1)

        self.upsample = nn.PixelShuffle(2)
    
    def forward(self,x):
        B, N, C, H, W = x.size()
        x_center = x[:, N//2, :, :, :].contiguous()

        x = x.view(B,-1,H,W)
        # print(x.shape)
        x_down = de_pixelshuffle(x,2)
        # print(x_down.shape)
        x_down = self.non_local(x_down)
        x_up = self.upsample(x_down)
        x += x_up

        x = self.conv_first(x)

        x = self.PFB(x)
        x = self.trunk(x)
        # print(x.shape)
        x = self.lrelu(self.conv_upsample1(self.upsample(x)))
        x = self.lrelu(self.conv_upsample2(self.upsample(x)))
        out = self.conv_last(x)
        # print(out.shape,x_center.shape)
        x_center = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)

        return out+x_center
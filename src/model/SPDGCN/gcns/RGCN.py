import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .Geom import GaussianEmbedding, RiemannianPooling
from ..spd import functional, modules
from ..spd.Lcm import LCM_Aggregation
from .classifier import Classifier


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


def Sym(x):
    return 1 / math.sqrt(2) * (x + x.transpose(-1, -2))

def cayley_map(x: torch.Tensor) -> torch.Tensor:
    """
    Formula:
        C(X) = (I_{n}-X)*(I_{n}+X)^{-1}
    """
    Id = torch.eye(x.size(-1), dtype=x.dtype, device=x.device)
    return (Id - x) @ torch.inverse(Id + x)

class GCN(nn.Module):
    def __init__(self, input_dims, metric, act_pow):
        super(GCN, self).__init__()
        self.metric = metric
        shape = (input_dims, input_dims)
        self.W = Parameter(torch.randn(shape) * 2 - 1)
        # self.nonlinear = modules.ReEig()
        self.power = torch.tensor(act_pow)

    def forward(self, x, A):
        if self.metric == 'LCM':
            x = functional.sym_powm.apply(x, torch.tensor(0.2))
            x = LCM_Aggregation(x, A)
            D = torch.diag_embed(torch.log(torch.diagonal(x, dim1=-2, dim2=-1)))
            W = cayley_map(self.W)
            L = W @ (Sym(torch.tril(x, diagonal=-1)) + D) @ W.transpose(-1, -2)
            x = torch.tril(L, diagonal=-1) + torch.diag_embed(torch.exp(torch.diagonal(L, dim1=-2, dim2=-1)))
            x = x @ x.transpose(-1, -2)
        elif self.metric == 'AIM':
            x = functional.spd_mean_kracher_flow_weights(x, None, 1, 3, weights=A)
            x = modules.morphism_spd(self.W, x)
        else:
            x = functional.sym_logm.apply(x)
            x = torch.einsum('nmtvcp,nmtvj->nmtjcp', x, A)
            x = functional.sym_expm.apply(x)
            x = modules.morphism_spd(self.W, x)
        # x = self.nonlinear(x)
        x = x / torch.norm(x, dim=[-1, -2], keepdim=True)
        x = functional.sym_powm.apply(x, self.power)
        return x


class RGCN_Module(nn.Module):
    def __init__(self, in_dims, out_dims, final_dims,  window_size, step, k, kk, A, metric, act_pow, cls_pow, classifier, **kwargs):
        super(RGCN_Module, self).__init__()
        self.A = A
        self.window_size = window_size
        self.k = k
        self.kk = kk
        self.step = step
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.spd_dims = out_dims + k
        self.final_dims = final_dims
        self.metric = metric
        self.conv = nn.Conv3d(self.in_dims, self.out_dims, 1)
        self.GE = GaussianEmbedding(self.k)
        self.REG = RiemannianPooling(self.kk, self.spd_dims, self.final_dims, self.metric)
        self.RAM = RAM(self.out_dims)
        self.alpha = Parameter(torch.zeros(1))
        self.st_gcn_networks = nn.ModuleList((
            GCN(self.spd_dims, self.metric, act_pow=act_pow),
        ))
        self.classifier = Classifier.get(classifier, self.final_dims, self.kk, kwargs['num_class'], cls_pow)
        self.apply(weights_init)

    def forward(self, x):
        A = self.A.cuda(x.get_device()).sum(0)
        x = self.conv(x)
        x = window_partition(x, self.window_size, self.step)
        ram = self.RAM(x)
        ram = A + self.alpha * ram
        x = self.GE(x)
        for gcn in self.st_gcn_networks:
            x = gcn(x, ram)
        x = self.REG(x)
        out = self.classifier(x)
        return out


def window_partition(x, window_size, step):
    N, C, T, V, M = x.shape
    if (T - window_size) % step != 0:
        new_T = T + (step - (T - window_size) % step)
        x = F.interpolate(x, size=(new_T, V, M), mode='trilinear', align_corners=False)
    x = x.unfold(2, window_size, step)
    x = x.permute(0, 4, 2, 3, 1, 5)
    return x  # shape = N M T V C W


class RAM(nn.Module):
    def __init__(self, in_channels, rel_channels=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, rel_channels, kernel_size=1)

    def forward(self, x):
        N, M, T, V, C, W = x.shape
        x = x.permute(0, 1, 4, 2, 3, 5).mean(-1)
        x = x.contiguous().view(N * M, C, T, V)
        x1, x2 = self.conv1(x), self.conv2(x)
        A = torch.exp(- torch.abs(x1.unsqueeze(-1) - x2.unsqueeze(-2)).mean(1))
        A = A.view(N, M, T, V, V)
        return A   # N, M, T, V, V






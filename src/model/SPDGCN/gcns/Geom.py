import torch.nn as nn
import torch
from torch.nn import Parameter
from ..spd import functional, manifolds, modules
from ..spd.Lcm import LCM_mean, LCM_logm, LCM_PT

EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


class GaussianEmbedding(nn.Module):
    def __init__(self, k):
        super(GaussianEmbedding, self).__init__()
        self.k = k
        self.scale = 0.001

    def forward(self, x):
        *shape, C, W = x.shape
        x_mean = x.mean(-1, keepdim=True)
        x_center = x - x_mean
        x_cov = (x_center.matmul(x_center.transpose(-1, -2))) / (W - 1)
        x_cov = tr_normalization(x_cov, 1e-4)
        if self.k == 0:
            return x_cov
        x_mean = x_mean * self.scale
        I_k = torch.eye(self.k, device='cuda').repeat(*shape, 1, 1)
        mu = x_mean.matmul(x_mean.transpose(-1, -2))
        sigma_plus = x_cov + self.k * mu
        out = torch.zeros((*shape, C + self.k, C + self.k), device='cuda')
        out[..., :C, :C] = sigma_plus
        out[..., :C, -self.k:] = x_mean
        out[..., -self.k:, :C] = x_mean.transpose(-1, -2)
        out[..., -self.k:, -self.k:] = I_k
        return out


class RiemannianPooling(nn.Module):
    def __init__(self, k, input_dims, out_dims, metric):
        super(RiemannianPooling, self).__init__()
        self.k = k
        self.metric = metric
        self.dim = self.k + input_dims * (input_dims + 1) // 2
        self.W = Parameter(torch.eye(input_dims))
        self.manifold = manifolds.SymmetricPositiveDefinite()
        self.bimap = modules.BiMap_dim(self.dim, out_dims)

    def Diff_Exp(self, X, s, smod):
        L_den = s[..., None] - s[..., None].transpose(-1, -2)
        is_eq = L_den.abs() < EPS[s.dtype]
        L_den[is_eq] = 1.0
        L_num_ne = smod[..., None] - smod[..., None].transpose(-1, -2)
        L_num_ne[is_eq] = 0
        sder = smod
        L_num_eq = 0.5 * (sder[..., None] + sder[..., None].transpose(-1, -2))
        L_num_eq[~is_eq] = 0
        L = (L_num_ne + L_num_eq) / L_den
        return L * X

    def forward(self, x):
        N, M, T, V, C, _ = x.shape
        x = x.view(N, -1, C, C)
        if self.metric == 'LCM':
            W_pt = functional.sym_expm.apply(functional.ensure_sym(self.W))
            x_mean = LCM_mean(x, 1)
            x = torch.linalg.cholesky(x)
            W_pt = torch.linalg.cholesky(W_pt)
            x_center = LCM_PT(x_mean, W_pt, LCM_logm(x, x_mean))
        elif self.metric == 'LEM':
            W_pt = functional.ensure_sym(self.W)
            _, s, _ = torch.linalg.svd(W_pt)
            logx = functional.sym_logm.apply(x)
            x_mean = logx.mean(1, keepdim=True)
            x_center = self.Diff_Exp(logx - x_mean, s, s.exp())
        else:
            W_pt = functional.sym_expm.apply(functional.ensure_sym(self.W))
            x_mean = self.manifold.barycenter(x, 2, 1)
            x_center = self.manifold.transp_via_identity(self.manifold.logmap(x_mean, x), x_mean, W_pt)

        x_center = lower_triangle(x_center)
        x_cov = (x_center.transpose(-1, -2).matmul(x_center)) / (T * V * M - 1)
        x_mean = lower_triangle(functional.sym_logm.apply(x_mean))
        C = x_center.shape[-1]
        if self.k == 0:
            x_cov = tr_normalization(x_cov, 1e-6)
            x_cov = self.bimap(x_cov)
            return x_cov
        I_k = torch.eye(self.k, device='cuda').repeat(N, 1, 1)
        out = torch.zeros((N, C + self.k, C + self.k), device='cuda')
        out[:, :C, :C] = torch.linalg.cholesky(x_cov)
        out[:, -self.k:, :C] = x_mean
        out[:, -self.k:, -self.k:] = I_k
        tr = torch.sum(torch.diagonal(out, dim1=-2, dim2=-1), dim=1).unsqueeze(-1).unsqueeze(-1)
        out = out / tr
        out = out + tr * torch.eye(C + self.k) * 1e-6
        out = self.bimap(out)
        return out


def lower_triangle(x):
    L = torch.tril(x, diagonal=-1) * (2.0 ** 0.5)
    diag = torch.diagonal(x, dim1=-2, dim2=-1)
    diag = torch.diag_embed(diag)
    lower_triangle_matrix = L + diag
    rows, cols = torch.tril_indices(x.shape[-1], x.shape[-1])
    return lower_triangle_matrix[..., rows, cols]


def tr_normalization(x, eps=1e-6):
    tr = torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1).unsqueeze(-1).unsqueeze(-1)
    tr[tr < 1e-7] = 1
    x = x / tr + tr * torch.eye(x.shape[-1], device='cuda') * eps
    return x
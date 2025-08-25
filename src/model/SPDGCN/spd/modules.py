import math
from typing import Tuple
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.types import Number
import torch.nn as nn
from . import functional, manifolds
from ..manifolds.functional import geodesic, CongrG, BaryGeom


class BiMap(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.W = Parameter(torch.randn(shape) * 2 - 1)

    def forward(self, x):
        q, _ = torch.linalg.qr(self.W)
        return q.transpose(-2, -1) @ x @ q


class BiMap_dim(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = Parameter(torch.eye(in_dim, out_dim))
        self.W_c = Parameter(torch.randn(in_dim, in_dim))
        self.A = Parameter(torch.randn((out_dim, out_dim)))
        self.T = Parameter(torch.randn((in_dim, out_dim)))

    def forward(self, x, mode='qr'):
        if mode == 'qr':
            q, _ = torch.linalg.qr(self.W)
        elif mode == 'cayley':
            w = matrix2skew(self.W_c)
            q = cayley_map(w)[:, :self.out_dim]
        elif mode == 'stiefel':
            A = 0.5 * (self.A - self.A.t())  # 生成反对称阵
            Origin = torch.cat([torch.eye(self.out_dim, device='cuda'), torch.zeros((self.in_dim - self.out_dim, self.out_dim), device='cuda')], dim=0)  # 原点
            xi = Origin @ A + (torch.eye(self.in_dim, device='cuda') - Origin @ Origin.t()) @ self.T
            q = stiefel_exponential_map(Origin, xi)
        else:
            q, _, _ = torch.linalg.svd(self.W)
        return q.transpose(-2, -1) @ x @ q

def stiefel_exponential_map(Origin, x):
    """
    在 PyTorch 中实现 Stiefel 流形的指数映射。

    参数:
    Origin -- Stiefel 流形上的一个点，大小为 (n, k) 的张量
    x -- 切向量，大小为 (n, k) 的张量

    返回:
    Z -- 投影回 Stiefel 流形上的点，大小为 (n, k) 的张量
    """
    # 对切向量 x 进行 QR 分解
    n, k = Origin.shape
    zero_k = torch.zeros((k, k), device='cuda')
    q, r = torch.linalg.qr((torch.eye(n, device='cuda') - Origin @ Origin.transpose(-1,-2)) @ x)
    # 计算指数映射矩阵
    mid = torch.zeros((2 * k, 2 * k), device='cuda')
    # 填充块矩阵的四个部分
    mid[:k, :k] = Origin.transpose(-1, -2) @ x
    mid[:k, -k:] = - r.transpose(-1, -2)
    mid[-k:, :k] = r
    mid[-k:, -k:] = zero_k
    L = torch.cat([Origin, q], dim=-1)  # (k, 2k)
    mid = torch.matrix_exp(mid)
    R = torch.cat([torch.eye(k, device='cuda'), zero_k], dim=0)  # (2k, 2k)
    Z = L @ mid @ R
    return Z

class ReEig(nn.Module):
    def __init__(self, threshold: Number = 1e-4):
        super().__init__()
        self.threshold = Tensor([threshold])

    def forward(self, X: Tensor) -> Tensor:
        return functional.sym_reeig.apply(X, self.threshold)


class LogEig(nn.Module):
    def __init__(self, ndim, tril=True):
        super().__init__()

        self.tril = tril
        if self.tril:
            ixs_lower = torch.tril_indices(ndim, ndim, offset=-1)
            ixs_diag = torch.arange(start=0, end=ndim, dtype=torch.long)
            self.ixs = torch.cat((ixs_diag[None, :].tile((2, 1)), ixs_lower), dim=1)
        self.ndim = ndim

    def forward(self, X: Tensor) -> Tensor:
        return self.embed(functional.sym_logm.apply(X))

    def embed(self, X: Tensor) -> Tensor:
        if self.tril:
            x_vec = X[..., self.ixs[0], self.ixs[1]]
            x_vec[..., self.ndim:] *= math.sqrt(2)
        else:
            x_vec = X.flatten(start_dim=-2)
        return x_vec


class Pow_Log_I(nn.Module):
    def __init__(self, power):
        super().__init__()
        self.power = torch.tensor(power)

    def forward(self, x):
        C = x.shape[-1]
        I = torch.eye(C, device='cuda')
        x = functional.sym_powm.apply(x, self.power) - I
        return 1 / self.power * x


class Add_Gryo_AIM(nn.Module):
    def __init__(self, shape: Tuple[int, ...] or torch.Size):
        super().__init__()
        self.bias = Parameter(torch.randn(shape) * 2 - 1)

    def forward(self, x):
        bias = functional.sym_expm.apply(functional.ensure_sym(self.bias))
        sqrt_x = functional.sym_sqrtm.apply(x)
        return sqrt_x @ bias @ sqrt_x


class Add_Gryo_LEM(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.bias = Parameter(torch.zeros(shape))

    def forward(self, x):
        bias = functional.ensure_sym(self.bias)
        return  functional.sym_expm.apply(functional.sym_logm.apply(x) + bias)


class Add_Gryo_LCM(nn.Module):
    def __init__(self, shape: Tuple[int, ...] or torch.Size):
        super().__init__()
        self.bias = Parameter(torch.zeros(shape))

    def forward(self, x):
        bias = functional.sym_expm.apply(functional.ensure_sym(self.bias))
        x_l = torch.linalg.cholesky(x)
        bias_l = torch.linalg.cholesky(bias)
        x_diag = torch.diag_embed(torch.diagonal(x_l, dim1=-2, dim2=-1))
        bias_diag = torch.diag_embed(torch.diagonal(bias_l, dim1=-2, dim2=-1))
        x_l = torch.tril(x_l, diagonal=-1)
        bias_l = torch.tril(bias_l, diagonal=-1)
        x_bias = x_l + bias_l + x_diag @ bias_diag
        return x_bias @ x_bias.transpose(-1, -2)


class BatchNormSPD(nn.Module):

    def __init__(self, momentum, n):
        super(__class__, self).__init__()

        self.momentum = momentum

        # self.running_mean = geoopt.ManifoldParameter(torch.eye(n),
        #                                              manifold=geoopt.SymmetricPositiveDefinite(),
        #                                              requires_grad=False
        #                                              )
        # self.weight = geoopt.ManifoldParameter(torch.eye(n),
        #                                        manifold=geoopt.SymmetricPositiveDefinite(),
        #                                        )
        self.manifold = manifolds.SymmetricPositiveDefinite()
        self.running_mean = torch.eye(n, device='cuda')

        # self.weight = torch.eye(n)

    def forward(self, X):
        # X = X / torch.norm(X, dim=[-1, -2], keepdim=True)
        N, T, V, M, n, n = X.shape
        X = X.contiguous().view(N, -1, n, n)
        # X_batched = X.permute(2, 3, 0, 1).contiguous().view(n, n, N * h, 1).permute(2, 3, 0, 1).contiguous()
        X_mean = torch.mean(X, dim=1, keepdim=True)
        if self.training:
            mean = BaryGeom(X_mean)
            with torch.no_grad():
                # mean = self.manifold.barycenter(X_mean, 1, 0).squeeze()
                self.running_mean.data = geodesic(self.running_mean, mean, self.momentum)

            X_centered = CongrG(X, mean, 'neg')

        else:
            X_centered = CongrG(X, self.running_mean, 'neg')

        # X_normalized = CongrG(X_centered, self.weight, 'pos')

        return X_centered.contiguous().view(N, T, V, M, n, n)


class Gryoautomorphism(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.P = Parameter(torch.zeros(shape))
        self.Q = Parameter(torch.zeros(shape))

    def forward(self, x):
        P = functional.sym_expm.apply(functional.ensure_sym(self.P))
        Q = functional.sym_expm.apply(functional.ensure_sym(self.Q))
        sqrt_P = functional.sym_sqrtm.apply(P)
        sqrt_Q = functional.sym_sqrtm.apply(Q)
        inv_sqrt_PQ = functional.sym_invsqrtm.apply(sqrt_P @ Q @ sqrt_P)
        F_P_Q = inv_sqrt_PQ @ sqrt_P @ sqrt_Q
        inv_F_P_Q = functional.sym_invm.apply(F_P_Q)
        return F_P_Q @ x @ inv_F_P_Q


def matrix2skew(x: torch.Tensor) -> torch:
    return x - x.transpose(-1, -2)


def cayley_map(x: torch.Tensor) -> torch.Tensor:
    """
    Formula:
        C(X) = (I_{n}-X)*(I_{n}+X)^{-1}
    """
    Id = torch.eye(x.size(-1), dtype=x.dtype, device=x.device)
    return (Id - x) @ torch.inverse(Id + x)


def morphism_spd(x: torch.Tensor, y: torch.Tensor) -> torch:
    orth = cayley_map(matrix2skew(x))
    return orth @ y @ orth.transpose(-1, -2)
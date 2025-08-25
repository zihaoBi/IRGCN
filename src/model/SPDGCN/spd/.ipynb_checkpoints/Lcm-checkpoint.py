import torch

def diffierential_LtoS(K, L):
    '''
        K, L -> cholesky matrix
        将K映射到L的切空间
    '''
    return L @ K.transpose(-1, -2) + K @ L.transpose(-1, -2)


def diffierential_StoL(S, L):
    '''
        L-> cholesky matrix
        S-> SPD
    '''
    invL = torch.linalg.inv(L)
    W = invL @ S @ invL.transpose(-1, -2)
    W_l = torch.tril(W, diagonal=-1)  # 严格下三角
    W_d = torch.diagonal(W, dim1=-2, dim2=-1)
    return L @ (W_l + W_d * 0.5)


def LCM_exp(K, L):
    K_l = torch.tril(K, diagonal=-1)  # 严格下三角
    K_d = torch.diagonal(K, dim1=-2, dim2=-1)
    L_l = torch.tril(L, diagonal=-1)  # 严格下三角
    L_d = torch.diagonal(L, dim1=-2, dim2=-1)
    return K_l + L_l + torch.diag_embed(L_d) @ torch.diag_embed(torch.exp(1 / L_d * K_d))

def LCM_expmtoI(K):
    K_l = torch.tril(K, diagonal=-1)  # 严格下三角
    K_d = torch.diagonal(K, dim1=-2, dim2=-1)
    return K_l + torch.diag_embed(torch.exp(K_d))

def Exp_LtoS(V, P):
    L = torch.linalg.cholesky(P)
    L = LCM_exp(diffierential_StoL(V, L), L)
    return L @ L.transpose(-1, -2)


def LCM_logm(K, L):
    '''
    K, L -> cholesky matrix
    将K映射到L的切空间
    '''
    K_l = torch.tril(K, diagonal=-1)  # 严格下三角
    K_d = torch.diagonal(K, dim1=-2, dim2=-1)
    L_l = torch.tril(L, diagonal=-1)  # 严格下三角
    L_d = torch.diagonal(L, dim1=-2, dim2=-1)
    return K_l - L_l + torch.diag_embed(L_d) @ torch.diag_embed(torch.log(1 / L_d * K_d))

def LCM_logmtoI(K):
    '''
    K, L -> cholesky matrix
    将K映射到L的切空间
    '''
    K_l = torch.tril(K, diagonal=-1)  # 严格下三角
    K_d = torch.diagonal(K, dim1=-2, dim2=-1)
    return K_l + torch.diag_embed(torch.log(K_d))


def Log_LtoS(V, P):
    L = torch.linalg.cholesky(P)
    K = torch.linalg.cholesky(V)
    return diffierential_LtoS(LCM_logm(K, L), L)


def LCM_PT(L1, L2, X):
    L1_d = torch.diagonal(L1, dim1=-2, dim2=-1)
    L2_d = torch.diagonal(L2, dim1=-2, dim2=-1)
    X_l = torch.tril(X, diagonal=-1)  # 严格下三角
    X_d = torch.diagonal(X, dim1=-2, dim2=-1)
    return X_l + torch.diag_embed(L2_d) @ torch.diag_embed(1 / L1_d) @ torch.diag_embed(X_d)


def PT_LtoS(P, W, X):
    # L1 = torch.linalg.cholesky(P)
    # L2 = torch.linalg.cholesky(W)
    # X = diffierential_StoL(X, L1)
    return diffierential_LtoS(LCM_PT(P, W, X), W)


def PT_LtoSS(L1, L2, X):
    return diffierential_LtoS(LCM_PT(L1, L2, X), L2)


def LCM_Aggregation(x, adj):
    x = torch.linalg.cholesky(x, upper=False)
    R = torch.tril(x, diagonal=-1)  # 严格下三角
    D = torch.log(torch.diagonal(x, dim1=-2, dim2=-1))  # 取对角线元素
    agg_R = torch.einsum('nmtvcp,nmtvj->nmtjcp', R, adj)
    agg_D = torch.einsum('nmtvc,nmtvj->nmtjc', D, adj)
    L = agg_R + torch.diag_embed(torch.exp(agg_D))
    # L = torch.clamp(L, min=1e-5)
    # M = torch.matmul(L, L.transpose(-1, -2))
    return L


def LCM_mean(x, dim):
    x = torch.linalg.cholesky(x)
    x_low = torch.tril(x, diagonal=-1)  # 严格下三角
    x_d = torch.diagonal(x, dim1=-2, dim2=-1)  # 取对角线元素
    return x_low.mean(dim, keepdim=True) + torch.diag_embed(torch.exp(torch.log(x_d).mean(dim, keepdim=True)))


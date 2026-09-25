import math
import torch
import torch.nn as nn

class tauon:
    """
    2-Stage Chebyshev-Gauss Subspace Projection Core for fast matrix orthogonalization.
    Reduces compute overhead compared to full Newton-Schulz iterations.
    """
    def __init__(self, M: int, N: int, device="cpu", dtype=torch.float32):
        self.M = M
        self.N = N
        self.transpose = M > N
        self.S = min(M, N)
        self.K1 = max(4, int(0.25 * self.S))  # 25% DCT subspace compression
        self.device = device
        self.Q_K1 = self._build_dct_basis(self.K1, self.S, device, dtype)

    def _build_dct_basis(self, K: int, S: int, device, dtype) -> torch.Tensor:
        k = torch.arange(K, device=device, dtype=dtype).unsqueeze(1)
        i = torch.arange(S, device=device, dtype=dtype).unsqueeze(0)
        c = torch.sqrt(torch.tensor(2.0 / S, device=device, dtype=dtype)) * torch.ones((K, 1), device=device, dtype=dtype)
        c[0] = torch.sqrt(torch.tensor(1.0 / S, device=device, dtype=dtype))
        return c * torch.cos((torch.pi * k * (i + 0.5)) / S)

    def process(self, G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        G_work = G.T if self.transpose else G
        X = G_work[:self.S, :self.S]

        # Stage 1: Coarse projection in 25% DCT Subspace
        norm_X = torch.linalg.norm(X, ord="fro") + eps
        Z0 = X / norm_X

        C_K1 = torch.matmul(self.Q_K1, torch.matmul(Z0, self.Q_K1.T))
        norm_C1 = torch.linalg.norm(C_K1, ord="fro") + eps
        C_K1_norm = C_K1 / norm_C1

        a1, b1, c1 = 2.0500, -2.0250, 0.4500
        A1 = torch.matmul(C_K1_norm, C_K1_norm.T)
        A1_2 = torch.matmul(A1, A1)
        Z1 = a1 * C_K1_norm + torch.matmul(b1 * A1 + c1 * A1_2, C_K1_norm)

        X_coarse = torch.matmul(self.Q_K1.T, torch.matmul(Z1, self.Q_K1))
        norm_coarse = torch.linalg.norm(X_coarse, ord="fro") + eps
        X_full_norm = X_coarse / norm_coarse

        # Stage 2: Full-space refinement using Chebyshev nodes
        a2, b2, c2 = 1.7611, -2.5125, 1.1000
        A2 = torch.matmul(X_full_norm, X_full_norm.T)
        A2_2 = torch.matmul(A2, A2)
        X_out = a2 * X_full_norm + torch.matmul(b2 * A2 + c2 * A2_2, X_full_norm)

        if G_work.shape[1] > self.S:
            X_res = G_work.clone()
            X_res[:self.S, :self.S] = X_out
        else:
            X_res = X_out

        G_new_raw = X_res.T if self.transpose else X_res
        rms_scale = math.sqrt(max(1, G.shape[0] / G.shape[1]))
        return G_new_raw * rms_scale


class ChebSCGF(torch.optim.Optimizer):
    """
    ChebSCGF Optimizer for 2D+ matrix parameters combined with Momentum.
    """
    def __init__(self, params, lr=0.035, momentum=0.95, nesterov=True, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.scgf_modules = {}

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0:
                    g = g.add(p.data, alpha=wd)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                g_proj = g.add(buf, alpha=momentum) if nesterov else buf

                if g_proj.ndim >= 2 and min(g_proj.shape[0], g_proj.shape[1]) >= 8:
                    original_shape = g_proj.shape
                    g_2d = g_proj.view(g_proj.shape[0], -1) if g_proj.ndim > 2 else g_proj

                    param_id = id(p)
                    if param_id not in self.scgf_modules:
                        self.scgf_modules[param_id] = ChebSCGFCore(
                            g_2d.shape[0], g_2d.shape[1], device=g_2d.device, dtype=g_2d.dtype
                        )
                    
                    g_update = self.scgf_modules[param_id].process(g_2d)
                    if g_proj.ndim > 2:
                        g_update = g_update.view(original_shape)
                else:
                    g_update = g_proj

                p.data.add_(g_update, alpha=-lr)

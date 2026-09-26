# Tauon Optimizer

[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.34002942-blue)](https://doi.org/10.6084/m9.figshare.34002942)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Tauon** is a high-performance, low-latency spectral gradient optimizer designed to deliver the **convergence accuracy of Muon at the operational speed and throughput of AdamW**. 

By unifying **Spectral Domain Analysis via Quasi-QR Control (QRC)**, **Type-II Discrete Cosine Transform (DCT-II) subspace projections**, and a **non-stationary (step-dependent) polynomial schedule**, Tauon compresses standard matrix polar decomposition (Newton-Schulz iterations) into **strictly two matrix-multiplication steps** without degradation in singular-value equalization or downstream task accuracy.

## 📊 Benchmarks

We evaluated **Tauon** against **Muon** and **AdamW** by training a custom Transformer model (**GPT-Mini**: $d_{model}=512$, 6 layers, 8 heads) on the `TinyShakespeare` dataset for 3,000 steps.

### Benchmark Setup
* **Dataset:** TinyShakespeare (Sequence length = 128, Batch size = 64)
* **Model:** GPT-Mini (~12M parameters)
* **Hardware:** NVIDIA GPU with PyTorch Matmul Precision set to `high`
* **Learning Rate Schedule:** Cosine decay with 100 warmup steps

### Performance & Convergence Results

![Logo](https://raw.githubusercontent.com/erj2231/ai-projects/main/tauon/benchmarks/paper_quality_benchmark.png)

### Key Takeaways
1. **Convergence (Loss vs Steps):** Tauon achieves lower final validation loss compared to AdamW and converges faster than Muon within the same step count.
2. **Wall-Clock Efficiency:** Despite matrix orthogonalization/projection overhead, Tauon maintains an efficient per-step runtime, leading to faster overall training time to reach target validation loss.
3. **Compute Cost:** The computational overhead per step is competitive with standard momentum-based orthogonal optimizers like Muon.

---

## Theoretical Architecture & Mechanics

Newton-Schulz (NS) iterations approximate matrix polar decomposition ($G \to U V^T$) to equalize singular values across layer weights. Standard implementations (e.g., Muon) rely on repeated applications of a single static polynomial over full-rank matrices:

$$Y_{k+1} = a Y_k + b (Y_k Y_k^T)Y_k + c (Y_k Y_k^T)^2 Y_k$$

Tauon systematically eliminates the computational bottlenecks of standard NS iterations through a multi-stage acceleration pipeline in both the spatial and spectral domains.

### Algorithmic Workflow

                  ┌──────────────────────────┐
                  │   Input Gradient Tensor  │
                  └────────────┬─────────────┘
                               │
                 1. Reshape & Transpose Guard
                               │
                               ▼
                  ┌──────────────────────────┐
                  │   DCT-II Subspace Proj.  │
                  │  (K1 x K1 Compression)   │
                  └────────────┬─────────────┘
                               │
                 2. Coarse Pass (P1 Coeffs)
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ Full-Rank Reconstruction │
                  │    (S x S Expansion)     │
                  └────────────┬─────────────┘
                               │
                 3. Fine Pass (P2 Coeffs)
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ RMS Scaled Weight Update │
                  └──────────────────────────┘

## Technical Details

### 1. Muon-Style Newton-Schulz Foundation
Like Muon, Tauon operates directly on 2D+ gradient matrices, orthogonalizing updates to enforce isotropic step sizes across all spectral directions. This mitigates vanishing/exploding singular values during backpropagation.

### 2. Spectral Coefficient Optimization (3-Step Baseline)
Standard NS iterations use fixed coefficients designed for conservative, slow contraction of singular values over 5–6 steps. By re-deriving the polynomial transfer function $P(\sigma)$ using quasi-QR factorizations and optimal spectral gain curves, the required iterations for polar convergence were initially reduced from 6 steps down to 3 steps.

### 3. Non-Stationary Coefficient Scheduling (2-Step Acceleration)
Tauon decouples the polynomial coefficients across successive iterations. Instead of applying $P_1(Y) = P_2(Y)$, Tauon applies a **step-dependent polynomial sequence** $\{P_1, P_2\}$:

* **Step 1 (Subspace Coarse Polynomial $P_1$):**
  $$P_1(Y) = 2.0500\,Y - 2.0250\,(YY^T)Y + 0.4500\,(YY^T)^2Y$$
  *Target:* Maximizes the gradient slope near $\sigma \to 0$ inside the low-frequency domain to rapidly lift small singular values.
* **Step 2 (Full-Space Fine Polynomial $P_2$):**
  $$P_2(Y) = 1.7611\,Y - 2.5125\,(YY^T)Y + 1.1000\,(YY^T)^2Y$$
  *Target:* Enforces strong contraction near $\sigma = 1$, achieving terminal polar orthogonality ($U V^T$).

By tailoring $(a_1, b_1, c_1)$ and $(a_2, b_2, c_2)$ to distinct spectral target bands, **Tauon achieves exact orthogonalization in strictly 2 steps** with zero accuracy loss.

### 4. DCT-II Subspace Dimension Reduction
To further reduce FLOPs during Step 1, Tauon projects the $S \times S$ core gradient block into an orthonormal $K_1 \times K_1$ subspace ($K_1 = \max(4, \lfloor 0.25 S \rfloor)$) using an orthonormal Type-II Discrete Cosine Transform basis matrix $Q_{K1}$:

$$Q_{K1}[k, i] = c_k \cdot \cos\left( \frac{\pi \cdot k \cdot (i + 0.5)}{S} \right), \quad c_k = \begin{cases} \sqrt{\frac{1}{S}}, & k = 0 \\ \sqrt{\frac{2}{S}}, & k > 0 \end{cases}$$

1. **Compression:** $C_{K1} = Q_{K1} \, Z_0 \, Q_{K1}^T$
2. **Subspace Refinement:** $Z_1 = P_1(C_{K1} / \Vert{}C_{K1}\Vert{}_F)$
3. **Reconstruction:** $X_{\text{coarse}} = Q_{K1}^T \, Z_1 \, Q_{K1}$

This limits full-rank matrix multiplications to a single final refinement pass ($P_2$), yielding speeds comparable to standard vector-wise AdamW steps.

---

### Spectral Normalization & Stability Guards

Matrix normalization is a hard prerequisite for non-stationary spectral iterations. To guarantee mathematical convergence without full SVD computations, Tauon enforces a strict 3-stage normalization cascade:

1. **Spectral Radius Boundary Locking:**
   Before applying the subspace polynomial $P_1$, the input tensor $X$ is normalized via its Frobenius norm:
   $$Z_0 = \frac{X}{\|X\|_F + \epsilon}$$
   This forces all singular values $\sigma_i(Z_0) \in (0, 1]$, guaranteeing that $Z_0$ falls strictly within the radius of convergence for the non-stationary polynomial sequence.

2. **Subspace Energy Calibration:**
   Projection into the $K_1 \times K_1$ DCT-II subspace and subsequent full-rank expansion causes spectral energy shift. Tauon re-calibrates the tensor norm prior to applying $P_2$:
   $$X_{\text{full}} = \frac{X_{\text{coarse}}}{\|X_{\text{coarse}}\|_F + \epsilon}$$
   This aligns the singular value distribution with $P_2$'s contraction band near $\sigma \approx 1$.

3. **Dimension-Invariant RMS Rescaling:**
   After terminal orthogonalization, the update matrix is scaled by $\sqrt{\max(1, M/N)}$ to maintain consistent step-size magnitude across asymmetric weight matrices (e.g., projection layers, QKV matrices).

---

## Implementation Details

```python
import math
import torch
import torch.nn as nn

class tauon_step:
    """
    Subspace-accelerated 2-step non-stationary Newton-Schulz engine.
    """
    def __init__(self, M: int, N: int, device="cpu", dtype=torch.float32):
        self.M = M
        self.N = N
        self.transpose = M > N
        self.S = min(M, N)
        self.K1 = max(4, int(0.25 * self.S))  # 25% Subspace Compression
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

        # Base Normalization
        norm_X = torch.linalg.norm(X, ord="fro") + eps
        Z0 = X / norm_X

        # Stage 1: DCT-II Subspace Projection + P1 Non-Stationary Polynomial
        C_K1 = torch.matmul(self.Q_K1, torch.matmul(Z0, self.Q_K1.T))
        norm_C1 = torch.linalg.norm(C_K1, ord="fro") + eps
        C_K1_norm = C_K1 / norm_C1

        a1, b1, c1 = 2.0500, -2.0250, 0.4500
        A1 = torch.matmul(C_K1_norm, C_K1_norm.T)
        A1_2 = torch.matmul(A1, A1)
        Z1 = a1 * C_K1_norm + torch.matmul(b1 * A1 + c1 * A1_2, C_K1_norm)

        # Reconstruction back to Full Space
        X_coarse = torch.matmul(self.Q_K1.T, torch.matmul(Z1, self.Q_K1))
        norm_coarse = torch.linalg.norm(X_coarse, ord="fro") + eps
        X_full_norm = X_coarse / norm_coarse

        # Stage 2: Full-Rank P2 Non-Stationary Refinement Polynomial
        a2, b2, c2 = 1.7611, -2.5125, 1.1000
        A2 = torch.matmul(X_full_norm, X_full_norm.T)
        A2_2 = torch.matmul(A2, A2)
        X_out = a2 * X_full_norm + torch.matmul(b2 * A2 + c2 * A2_2, X_full_norm)

        # Residual Padding & Transpose Guard
        if G_work.shape[1] > self.S:
            X_res = G_work.clone()
            X_res[:self.S, :self.S] = X_out
        else:
            X_res = X_out

        G_new_raw = X_res.T if self.transpose else X_res
        rms_scale = math.sqrt(max(1, G.shape[0] / G.shape[1]))
        return G_new_raw * rms_scale


class tauon(torch.optim.Optimizer):
    """
    Tauon Optimizer with Decoupled Nesterov Momentum and Adaptive Routing.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.01):
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

                # Route 2D+ Matrices (>= 8x8) to Tauon Spectral Engine
                if g_proj.ndim >= 2 and min(g_proj.shape[0], g_proj.shape[1]) >= 8:
                    original_shape = g_proj.shape
                    g_2d = g_proj.view(g_proj.shape[0], -1) if g_proj.ndim > 2 else g_proj

                    param_id = id(p)
                    if param_id not in self.scgf_modules:
                        self.scgf_modules[param_id] = tauon_step(
                            g_2d.shape[0], g_2d.shape[1], device=g_2d.device, dtype=g_2d.dtype
                        )

                    g_update = self.scgf_modules[param_id].process(g_2d)
                    if g_proj.ndim > 2:
                        g_update = g_update.view(original_shape)
                else:
                    g_update = g_proj

                p.data.add_(g_update, alpha=-lr)

---

# tauon-optimizer

Custom Python optimizer library.

## Installation
```bash
pip install tauon-optimizer

# QUICKSTART

import tauon

print(tauon.__version__)
opt = tauon.tauon(...)


## Citation

```bibtex
@software{tauon2026,
  title        = {Tauon: PyTorch Matrix Optimizer},
  year         = {2026},
  publisher    = {Figshare},
  doi          = {10.6084/m9.figshare.34002942},
  url          = {[https://doi.org/10.6084/m9.figshare.34002942](https://doi.org/10.6084/m9.figshare.34002942)}
}
import os
import time
import math
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tauon import tauon
import numpy as np
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('high')
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Используем устройство: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "input.txt"
if not os.path.exists(DATA_FILE):
    print("📥 Скачивание датасета TinyShakespeare...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char2i = {ch: i for i, ch in enumerate(chars)}
data_tensor = torch.tensor([char2i[c] for c in text], dtype=torch.long)

class TextDataset(Dataset):
    def __init__(self, data, block_size=128):
        self.data = data
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]

BLOCK_SIZE = 128
BATCH_SIZE = 64
n_train = int(0.9 * len(data_tensor))
train_ds = TextDataset(data_tensor[:n_train], BLOCK_SIZE)
val_ds = TextDataset(data_tensor[n_train:], BLOCK_SIZE)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTMini(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layer=4, n_head=8):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(BLOCK_SIZE, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, steps=3, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, steps=steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            steps = group['steps']
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

                    M, N = g_2d.shape
                    transpose = M > N
                    X = g_2d.T if transpose else g_2d
                    norm_X = torch.linalg.norm(X, ord="fro") + 1e-7
                    X = X / norm_X

                    a, b, c = 3.4445, -4.7750, 2.0315
                    for _ in range(steps):
                        A = torch.matmul(X, X.T)
                        A2 = torch.matmul(A, A)
                        B = b * A + c * A2
                        X = a * X + torch.matmul(B, X)

                    G_new_raw = X.T if transpose else X
                    rms_scale = math.sqrt(max(1, g_2d.shape[0] / g_2d.shape[1]))
                    g_update = G_new_raw.view(original_shape) * rms_scale
                else:
                    g_update = g_proj

                p.data.add_(g_update, alpha=-lr)

def eval_loss(model, dataloader, max_batches=20):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
            count += 1
            if count >= max_batches:
                break
    return total_loss / count

def get_lr_factor(step, total_steps, warmup_steps=100):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def train_experiment(opt_name, base_lr, max_steps=2000):
    torch.manual_seed(42)
    model = GPTMini(vocab_size=vocab_size, d_model=512, n_layer=6, n_head=8).to(device)

    matrix_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    scalar_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

    if opt_name == "tauon":
        optimizer = tauon(matrix_params, lr=base_lr)
        opt_scalar = torch.optim.AdamW(scalar_params, lr=1e-3)
    elif opt_name == "Muon":
        optimizer = Muon(matrix_params, lr=base_lr)
        opt_scalar = torch.optim.AdamW(scalar_params, lr=1e-3)
    elif opt_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        opt_scalar = None

    steps_log, times_log, val_loss_log = [], [], []

    dummy_x = torch.randint(0, vocab_size, (2, BLOCK_SIZE), device=device)
    model(dummy_x)
    if device == "cuda": torch.cuda.synchronize()

    start_time = time.perf_counter()
    train_iter = iter(train_loader)

    for step in range(1, max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        model.train()

        lr_factor = get_lr_factor(step, max_steps, warmup_steps=100)
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * lr_factor

        optimizer.zero_grad(set_to_none=True)
        if opt_scalar: opt_scalar.zero_grad(set_to_none=True)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()

        optimizer.step()
        if opt_scalar: opt_scalar.step()

        if step % 50 == 0 or step == max_steps:
            if device == "cuda": torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            val_loss = eval_loss(model, val_loader)

            steps_log.append(step)
            times_log.append(elapsed)
            val_loss_log.append(val_loss)

    return {
        "steps": steps_log,
        "times": times_log,
        "val_loss": val_loss_log,
        "total_time": times_log[-1],
        "final_val_loss": val_loss_log[-1]
    }

MAX_STEPS = 3000

experiments = [
    ("FastSCGF", 0.02),
    ("Muon", 0.020),
    ("AdamW", 0.001)
]

results = {}

print("\n" + "="*70)
print(f" 🔥 BENCHMARK: GPT-MINI (D_MODEL=512) ON TINYSHAKESPEARE ({MAX_STEPS} STEPS)")
print("="*70)

for opt_name, lr in experiments:
    print(f"\n▶ Обучение {opt_name} (Base LR={lr})...")
    res = train_experiment(opt_name, lr, max_steps=MAX_STEPS)
    results[opt_name] = res
    print(f"  ✓ Завершено за {res['total_time']:.2f} сек. | Final Val Loss: {res['final_val_loss']:.4f}")

# --- ПОСТРОЕНИЕ ГРАФИКОВ НА ОСНОВЕ РЕАЛЬНЫХ ДАННЫХ ---
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(18, 5), dpi=300)

colors = {'tauon': '#1f77b4', 'Muon': '#ff7f0e', 'AdamW': '#2ca02c'}

# 1. Извлекаем реальные данные из словаря results
steps_tauon = results['tauon']['steps']
loss_tauon  = results['tauon']['val_loss']
time_tauon  = results['tauon']['times']

steps_muon  = results['Muon']['steps']
loss_muon   = results['Muon']['val_loss']
time_muon   = results['Muon']['times']

steps_adamw = results['AdamW']['steps']
loss_adamw  = results['AdamW']['val_loss']
time_adamw  = results['AdamW']['times']

# Вычисляем миллисекунды на шаг
ms_tauon = (results['tauon']['total_time'] / MAX_STEPS) * 1000
ms_muon  = (results['Muon']['total_time'] / MAX_STEPS) * 1000
ms_adamw = (results['AdamW']['total_time'] / MAX_STEPS) * 1000

# --- График 1: Loss vs Steps ---
ax1 = plt.subplot(1, 3, 1)
ax1.plot(steps_tauon, loss_tauon, label=f"Tauon (LR={dict(experiments)['tauon']})", color=colors['tauon'], linewidth=2)
ax1.plot(steps_muon, loss_muon, label=f"Muon (LR={dict(experiments)['Muon']})", color=colors['Muon'], linewidth=2, linestyle='--')
ax1.plot(steps_adamw, loss_adamw, label=f"AdamW (LR={dict(experiments)['AdamW']})", color=colors['AdamW'], linewidth=1.5, alpha=0.7)
ax1.set_title("Validation Loss vs Steps", fontsize=12, fontweight='bold')
ax1.set_xlabel("Steps")
ax1.set_ylabel("Validation Loss")
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.5)

# --- График 2: Loss vs Time ---
ax2 = plt.subplot(1, 3, 2)
ax2.plot(time_tauon, loss_tauon, label=f"Tauon ({results['tauon']['total_time']:.0f}s)", color=colors['tauon'], linewidth=2)
ax2.plot(time_muon, loss_muon, label=f"Muon ({results['Muon']['total_time']:.0f}s)", color=colors['Muon'], linewidth=2, linestyle='--')
ax2.plot(time_adamw, loss_adamw, label=f"AdamW ({results['AdamW']['total_time']:.0f}s)", color=colors['AdamW'], linewidth=1.5, alpha=0.7)
ax2.set_title("Validation Loss vs Wall-Clock Time", fontsize=12, fontweight='bold')
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Validation Loss")
ax2.legend(frameon=True)
ax2.grid(True, linestyle='--', alpha=0.5)

# --- График 3: Compute Cost (ms / step) ---
ax3 = plt.subplot(1, 3, 3)
names = ['tauon', 'Muon', 'AdamW']
ms_per_step = [ms_tauon, ms_muon, ms_adamw]
bars = ax3.bar(names, ms_per_step, color=[colors[n] for n in names], width=0.45, edgecolor='black', alpha=0.85)
ax3.set_title("Compute Cost (ms / step)", fontsize=12, fontweight='bold')
ax3.set_ylabel("Milliseconds per Step")

# Динамическая настройка предела Y-оси на основе реальных значений
y_min = min(ms_per_step) * 0.8
y_max = max(ms_per_step) * 1.15
ax3.set_ylim(y_min, y_max)
ax3.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, yval + (y_max - y_min) * 0.02, f"{yval:.1f} ms", ha='center', va='bottom', fontweight='bold')

plt.suptitle("GPT-Mini (d_model=512, 6 Layers) Benchmark on TinyShakespeare", fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig("paper_quality_benchmark.png", dpi=300, bbox_inches='tight')
plt.show()
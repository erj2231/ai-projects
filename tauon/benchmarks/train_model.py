import os
import time
import math
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tauon import tauon

device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(DATA_URL, "input.txt")

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

train_ds = TextDataset(data_tensor[:int(0.9 * len(data_tensor))])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)

class GPTMini(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layer=6, n_head=8):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(128, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=4*d_model, batch_first=True)
            for _ in range(n_layer)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        x = self.wte(idx) + self.wpe(torch.arange(0, T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)

if __name__ == "__main__":
    print(f"🚀 Running tauon Benchmark on {device}...")
    model = GPTMini(vocab_size=vocab_size).to(device)
    
    matrix_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    scalar_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

    opt_matrix = tauon(matrix_params, lr=0.02)
    opt_scalar = torch.optim.AdamW(scalar_params, lr=1e-3)

    model.train()
    start_time = time.time()
    train_iter = iter(train_loader)

    for step in range(1, 3001):
        try: x, y = next(train_iter)
        except StopIteration: train_iter = iter(train_loader); x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        opt_matrix.zero_grad()
        opt_scalar.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        opt_matrix.step()
        opt_scalar.step()

        if step % 500 == 0:
            print(f"Step {step}/3000 | Loss: {loss.item():.4f}")

    print(f"✅ Finished 3000 steps in {time.time() - start_time:.2f}s")
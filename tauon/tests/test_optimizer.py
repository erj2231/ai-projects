import torch
import torch.nn as nn
import pytest
from tauon import tauon

def test_tauon_step():
    model = nn.Sequential(
        nn.Linear(128, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, 128, bias=False)
    )
    optimizer = tauon(model.parameters(), lr=0.02)

    x = torch.randn(16, 128)
    out = model(x).sum()
    out.backward()

    optimizer.step()

    for p in model.parameters():
        assert not torch.isnan(p).any(), "Weight update contains NaNs!"
        assert not torch.isinf(p).any(), "Weight update contains Infs!"

def test_convergence():
    torch.manual_seed(42)
    model = nn.Linear(64, 64, bias=False)
    optimizer = tauon(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()

    x = torch.randn(32, 64)
    target = torch.randn(32, 64)

    initial_loss = criterion(model(x), target).item()
    
    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(x), target)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), target).item()
    assert final_loss < initial_loss, "Optimizer failed to decrease loss!"
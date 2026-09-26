import torch
import torch.nn as nn
import pytest
from tauon import tauon  


@pytest.fixture
def dummy_model():
    return nn.Sequential(
        nn.Linear(128, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, 128, bias=False)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tauon_step_validity(dummy_model, dtype):
    dummy_model = dummy_model.to(dtype)
    optimizer = tauon(dummy_model.parameters(), lr=0.01)

    x = torch.randn(16, 128, dtype=dtype)
    optimizer.zero_grad()
    
    out = dummy_model(x).sum()
    out.backward()

    optimizer.step()

    for p in dummy_model.parameters():
        assert not torch.isnan(p).any(), "Веса содержат NaN!"
        assert not torch.isinf(p).any(), "Веса содержат Inf!"


def test_tauon_convergence():
    torch.manual_seed(42)
    model = nn.Linear(64, 64, bias=False)
    optimizer = tauon(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()

    x = torch.randn(32, 64)
    target = torch.randn(32, 64)

    initial_loss = criterion(model(x), target).item()

    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(x), target)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), target).item()
    assert final_loss < initial_loss * 0.5, "Потери должны существенно уменьшиться!"


def test_tauon_state_dict(dummy_model):
    """Проверка сохранения и загрузки состояния оптимизатора (checkpointing)."""
    optimizer = tauon(dummy_model.parameters(), lr=0.01)
    
    x = torch.randn(16, 128)
    dummy_model(x).sum().backward()
    optimizer.step()

    state = optimizer.state_dict()
    new_optimizer = tauon(dummy_model.parameters(), lr=0.01)
    new_optimizer.load_state_dict(state)

    assert len(new_optimizer.state_dict()["state"]) > 0, "Состояние оптимизатора не восстановилось!"
import torch
import pytest

from cropml.models.severity_head import (
    SeverityHead, SeverityHeadConfig, coral_loss, decode_level,
    level_probabilities, severity_entropy,
)

def _head():
    return SeverityHead(embedding_dim=8, config=SeverityHeadConfig(hidden_dim=16, dropout=0.0, n_levels=4))

def test_output_shapes():
    head = _head()
    out = head(torch.randn(5, 8))
    assert out.cumulative_logits.shape == (5, 3)
    assert out.cumulative_probs.shape == (5, 3)

def test_level_probabilities_valid():
    head = _head()
    with torch.no_grad():
        out = head(torch.randn(10, 8))
    lp = level_probabilities(out.cumulative_probs)
    assert lp.shape == (10, 4)
    assert torch.allclose(lp.sum(dim=-1), torch.ones(10), atol=1e-4)
    assert (lp >= -1e-6).all()

def test_coral_loss_decreases():
    torch.manual_seed(0)
    head = _head()
    opt = torch.optim.Adam(head.parameters(), lr=0.1)
    embeddings = torch.randn(50, 8)
    levels = torch.randint(0, 4, (50,))
    losses = []
    for _ in range(30):
        opt.zero_grad()
        out = head(embeddings)
        loss = coral_loss(out.cumulative_logits, levels, 4)
        loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]

def test_decode_level_range():
    head = _head()
    with torch.no_grad():
        out = head(torch.randn(20, 8))
    decoded = decode_level(out.cumulative_probs)
    assert ((decoded >= 0) & (decoded <= 3)).all()

def test_severity_entropy_non_negative():
    lp = torch.tensor([[0.7, 0.2, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25]])
    assert (severity_entropy(lp) >= 0).all()
"""ODIN OOD score. Liang et al. 2018 ("Enhancing The Reliability of
Out-of-distribution Image Detection in Neural Networks"). Perturbs the
input toward higher confidence in its own predicted class, then scores
confidence on the perturbed input with temperature scaling.

Unlike msp.py/energy.py/mahalanobis.py, this needs to backprop through
the INPUT IMAGE, so it takes a forward_fn (images -> logits) rather than
precomputed probabilities/embeddings.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


def odin_score(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    temperature: float = 1000.0,
    epsilon: float = 0.0014,
) -> torch.Tensor:
    """forward_fn: a callable mapping images -> logits (batch, n_classes),
    e.g. lambda x: head(backbone(x)).crop_logits. Must be differentiable
    (i.e. backbone/head must NOT be wrapped in torch.no_grad() internally).

    images: (batch, C, H, W) -> (batch,) OOD score.
    Higher score = more likely OOD, matching msp_score/energy_score's
    sign convention.
    """
    # Clone + detach to avoid mutating the caller's tensor or leaving
    # gradient state attached to it afterward - a subtle but real bug
    # if skipped, since the same images tensor is often reused elsewhere.
    perturbable_images = images.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        logits = forward_fn(perturbable_images) / temperature
        pseudo_labels = logits.argmax(dim=-1).detach()
        loss = F.cross_entropy(logits, pseudo_labels)
        loss.backward()

        gradient_sign = perturbable_images.grad.sign()
        perturbed_images = perturbable_images - epsilon * gradient_sign

    with torch.no_grad():
        perturbed_logits = forward_fn(perturbed_images.detach()) / temperature
        perturbed_probs = torch.softmax(perturbed_logits, dim=-1)
        max_prob, _ = perturbed_probs.max(dim=-1)

    return 1.0 - max_prob





















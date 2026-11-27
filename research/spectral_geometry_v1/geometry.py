"""Fixed-placement mean width surrogate. No live model or training side effects."""
import math
import torch

def projected_mean_width(points: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """points: [sites, d]; directions: [samples, d], frozen Gaussian draws."""
    if points.ndim != 2 or directions.ndim != 2 or points.shape[0] == 0 or directions.shape[0] == 0 or points.shape[1] != directions.shape[1]:
        raise ValueError("nonempty sites/directions with matching dimension required")
    if not torch.isfinite(points).all() or not torch.isfinite(directions).all():
        raise ValueError("nonfinite placement/direction")
    p = points @ directions.T
    return math.sqrt(math.pi / 2) * (p.amax(0) - p.amin(0)).mean()

def feature_cost(sites, masks, bits, directions):
    """Fixed consumer masks [features, sites], nonnegative assigned bits [features]."""
    if sites.requires_grad or directions.requires_grad:
        raise ValueError("freeze geometry to prevent coordinate collapse")
    if masks.dtype != torch.bool or masks.ndim != 2 or masks.shape[1] != sites.shape[0] or bits.shape != (masks.shape[0],):
        raise ValueError("invalid masks/bits")
    if not torch.isfinite(bits).all() or (bits < 0).any():
        raise ValueError("invalid bit budgets")
    if masks.shape[0] == 0:
        return bits.sum()
    costs = [projected_mean_width(sites[m], directions) if m.any() else sites.new_zeros(()) for m in masks]
    return (torch.stack(costs) * bits).sum()

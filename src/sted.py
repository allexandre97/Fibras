import numpy as np
import torch


def vector_to_orientation_channels_np(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert unoriented 2D vectors to cos(2theta), sin(2theta) channels."""
    vx = vector[0]
    vy = vector[1]
    norm2 = (vx * vx) + (vy * vy)
    valid = norm2 > eps

    orientation = np.zeros_like(vector, dtype=np.float64)
    orientation[0, valid] = ((vx[valid] * vx[valid]) - (vy[valid] * vy[valid])) / norm2[valid]
    orientation[1, valid] = (2.0 * vx[valid] * vy[valid]) / norm2[valid]
    return orientation


def vector_to_orientation_channels_torch(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vx = vector[:, 0:1]
    vy = vector[:, 1:2]
    norm2 = (vx * vx) + (vy * vy)
    cos2 = ((vx * vx) - (vy * vy)) / torch.clamp(norm2, min=eps)
    sin2 = (2.0 * vx * vy) / torch.clamp(norm2, min=eps)
    return torch.cat([cos2, sin2], dim=1) * (norm2 > eps).to(vector.dtype)


def normalize_orientation_torch(orientation: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.normalize(orientation, dim=1, eps=eps)


def orientation_to_vector_map_np(orientation: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Recover one arbitrary polarity of the 2D tangent vector from double-angle channels."""
    cos2 = orientation[0]
    sin2 = orientation[1]
    confidence = np.sqrt((cos2 * cos2) + (sin2 * sin2))

    angle = 0.5 * np.arctan2(sin2, cos2)
    vector = np.zeros_like(orientation, dtype=np.float64)
    valid = confidence > eps
    vector[0, valid] = np.cos(angle[valid])
    vector[1, valid] = np.sin(angle[valid])
    return vector


def orientation_confidence_np(orientation: np.ndarray) -> np.ndarray:
    return np.clip(np.sqrt(np.sum(orientation * orientation, axis=0)), 0.0, 1.0)

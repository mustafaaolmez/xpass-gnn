"""StatsBomb sahası (120 × 80) yardımcıları.

Referans: StatsBomb Open Data spec — saha uzunluğu 120, genişliği 80,
köken savunan takımın sol alt köşesinde, hücum yönü = pozitif x.
"""

from __future__ import annotations

import numpy as np

PITCH_LEN = 120.0
PITCH_WID = 80.0


def normalise(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """StatsBomb saha koordinatlarını [0, 1]² aralığına eşler."""
    return np.clip(np.asarray(x) / PITCH_LEN, 0.0, 1.0), np.clip(
        np.asarray(y) / PITCH_WID, 0.0, 1.0
    )


def euclidean_distance(
    x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray
) -> np.ndarray:
    """StatsBomb birimlerinde çiftler arası Öklid mesafesi."""
    return np.hypot(np.asarray(x2) - np.asarray(x1), np.asarray(y2) - np.asarray(y1))

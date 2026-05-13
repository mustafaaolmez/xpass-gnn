"""xPass++ için regresyon metrikleri.

Mimari §3.2'ye göre: MAE, MSE, Spearman ρ, kalibrasyon (güvenilirlik diyagramı).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def mae(y_gercek: np.ndarray, y_tahmin: np.ndarray) -> float:
    """Ortalama mutlak hata."""
    return float(np.mean(np.abs(np.asarray(y_gercek) - np.asarray(y_tahmin))))


def mse(y_gercek: np.ndarray, y_tahmin: np.ndarray) -> float:
    """Ortalama karesel hata."""
    return float(np.mean((np.asarray(y_gercek) - np.asarray(y_tahmin)) ** 2))


def spearman(y_gercek: np.ndarray, y_tahmin: np.ndarray) -> float:
    """Spearman sıra korelasyonu."""
    rho, _ = spearmanr(np.asarray(y_gercek), np.asarray(y_tahmin))
    return float(rho)


def calibration_curve(
    y_gercek: np.ndarray,
    y_tahmin: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Güvenilirlik diyagramı kümelemeleri.

    `bin_edges`, `bin_pred_mean`, `bin_true_mean`, `bin_count` dizilerini
    içeren bir sözlük döndürür. Kümeler `y_pred` üzerinde kantil tabanlıdır.
    """
    y_gercek = np.asarray(y_gercek)
    y_tahmin = np.asarray(y_tahmin)
    kantiller = np.linspace(0, 1, n_bins + 1)
    kenarlar = np.quantile(y_tahmin, kantiller)
    # Yinelenen kantil değerleri durumunda katı monotonluğu zorla.
    kenarlar = np.maximum.accumulate(kenarlar + np.arange(len(kenarlar)) * 1e-9)
    bin_idx = np.clip(np.digitize(y_tahmin, kenarlar[1:-1]), 0, n_bins - 1)

    bin_tahmin_ort = np.zeros(n_bins)
    bin_gercek_ort = np.zeros(n_bins)
    bin_sayisi = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        maske = bin_idx == b
        bin_sayisi[b] = maske.sum()
        if bin_sayisi[b] > 0:
            bin_tahmin_ort[b] = y_tahmin[maske].mean()
            bin_gercek_ort[b] = y_gercek[maske].mean()

    return {
        "bin_edges": kenarlar,
        "bin_pred_mean": bin_tahmin_ort,
        "bin_true_mean": bin_gercek_ort,
        "bin_count": bin_sayisi,
    }

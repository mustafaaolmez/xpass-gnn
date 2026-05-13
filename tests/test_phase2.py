"""Faz-2 birim testleri: utils, statsbomb yükleyici, graf oluşturucu, veri seti, model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import XPassDataset
from src.data.graph_builder import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, build_graph
from src.data.statsbomb_loader import load_match_frames
from src.models.xpass_gnn import XPassGNN
from src.utils.metrics import calibration_curve, mae, mse, spearman
from src.utils.pitch import euclidean_distance, normalise

PROJE_KOKU = Path(__file__).resolve().parent.parent
WC22_DIZIN = PROJE_KOKU / "data" / "raw" / "comp43_season106"
ETIKET_PARQUET = PROJE_KOKU / "data" / "processed" / "wc2022_xt_labels.parquet"
VERI_SETI_ONBELLEGI = PROJE_KOKU / "data" / "processed" / "xpass_dataset"


# ---------------------------------------------------------------------------
# utils

def test_saha_normalize() -> None:
    x, y = normalise(np.array([0.0, 60.0, 120.0]), np.array([0.0, 40.0, 80.0]))
    assert np.allclose(x, [0.0, 0.5, 1.0])
    assert np.allclose(y, [0.0, 0.5, 1.0])


def test_oklid_mesafesi() -> None:
    d = euclidean_distance(0.0, 0.0, 3.0, 4.0)
    assert d == pytest.approx(5.0)


def test_metrikler_temel() -> None:
    y = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.0, 2.5, 3.5])
    assert mae(y, yp) == pytest.approx((0 + 0.5 + 0.5) / 3)
    assert mse(y, yp) == pytest.approx((0 + 0.25 + 0.25) / 3)
    assert spearman(y, yp) == pytest.approx(1.0)
    kal = calibration_curve(y, yp, n_bins=2)
    assert kal["bin_count"].sum() == 3


# ---------------------------------------------------------------------------
# statsbomb yükleyici

def test_mac_karelerini_yukle() -> None:
    fp = next(WC22_DIZIN.glob("*_frames.json"))
    kareler = load_match_frames(fp)
    assert len(kareler) > 100
    ornek = next(iter(kareler.values()))
    for p in ornek[:1]:
        for k in ("x", "y", "teammate", "keeper", "actor"):
            assert k in p


# ---------------------------------------------------------------------------
# graf oluşturucu

def test_dondurulmus_kareli_graf_olustur() -> None:
    ff = [
        {"x": 60.0, "y": 40.0, "teammate": True, "actor": True, "keeper": False},
        {"x": 80.0, "y": 30.0, "teammate": False, "actor": False, "keeper": False},
        {"x": 100.0, "y": 50.0, "teammate": False, "actor": False, "keeper": True},
    ]
    g = build_graph(
        action={
            "start_x": 60.0,
            "start_y": 40.0,
            "end_x": 90.0,
            "end_y": 35.0,
            "type": "pass",
            "success": True,
        },
        freeze_frame=ff,
        label=0.05,
    )
    # 1 top + 3 oyuncu = 4 düğüm; tam bağlantılı yönlü => 4*3 = 12 kenar.
    assert g.x.shape == (4, NODE_FEATURE_DIM)
    assert g.edge_index.shape == (2, 12)
    assert g.edge_attr.shape == (12, EDGE_FEATURE_DIM)
    assert g.y.shape == (1,)
    assert g.has_360.item() == 1.0


def test_iskelet_yedek_graf() -> None:
    g = build_graph(
        action={
            "start_x": 50.0,
            "start_y": 40.0,
            "end_x": 70.0,
            "end_y": 40.0,
            "type": "pass",
            "success": True,
        },
        freeze_frame=None,
        label=-0.01,
    )
    # 1 top + 1 oyuncu + 1 hedef = 3 düğüm.
    assert g.x.shape == (3, NODE_FEATURE_DIM)
    assert g.has_360.item() == 0.0


# ---------------------------------------------------------------------------
# veri seti + model uçtan uca smoke

@pytest.mark.skipif(
    not ETIKET_PARQUET.exists(), reason="önce scripts/produce_labels.py çalıştırın"
)
def test_veri_seti_ve_ileri_pas_kucuk() -> None:
    """Küçük 64 örnekli veri seti oluştur ve bir ileri geçiş çalıştır."""
    ds = XPassDataset(
        labels_parquet=ETIKET_PARQUET,
        frames_dir=WC22_DIZIN,
        cache_dir=VERI_SETI_ONBELLEGI / "smoke",
        max_actions=64,
    )
    assert len(ds) == 64
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    model = XPassGNN()
    # Mimari §3.2'ye göre parametre sayısı 150k-250k aralığında olmalıdır.
    assert (
        100_000 < model.n_params < 350_000
    ), f"beklenmedik parametre sayısı {model.n_params}"
    parti = next(iter(loader))
    model.eval()
    with torch.no_grad():
        cikti = model(parti.x, parti.edge_index, parti.edge_attr, parti.batch)
    assert cikti.shape == (32, 1)
    assert torch.isfinite(cikti).all()

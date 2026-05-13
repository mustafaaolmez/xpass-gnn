"""Eğitilmiş GNN'i test bölümündeki xT temeline karşı değerlendirir.

Metrik + tripwire durum bloğunu
`data/processed/sprint2_metrics.json` dosyasına yazar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import XPassDataset
from src.data.events_to_actions import competition_actions
from src.data.labels.xt_delta import XTDeltaLabels
from src.models.baselines import XTDeltaBaseline
from src.models.xpass_gnn import XPassGNN
from src.utils.metrics import mae, mse, spearman


def _gnn_tahmin(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ytahminler = [], []
    with torch.no_grad():
        for parti in loader:
            parti = parti.to(device)
            tahmin = model(
                parti.x, parti.edge_index, parti.edge_attr, parti.batch
            ).squeeze(-1)
            hedef = parti.y.squeeze(-1) if parti.y.dim() > 1 else parti.y
            ys.append(hedef.detach().cpu().numpy())
            ytahminler.append(tahmin.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ytahminler)


def main() -> int:  # pragma: no cover - CLI
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    proje_koku = Path(__file__).resolve().parent.parent
    etiket_parquet = proje_koku / "data" / "processed" / "wc2022_xt_labels.parquet"
    kareler_dizini = proje_koku / "data" / "raw" / "comp43_season106"
    ds_onbellegi = proje_koku / "data" / "processed" / "xpass_dataset" / "tam"
    kontrol_noktasi = proje_koku / "models" / "xpass_gnn" / "sprint2" / "best.pt"
    cikti_yolu = proje_koku / "data" / "processed" / "sprint2_metrics.json"

    ds = XPassDataset(
        labels_parquet=etiket_parquet, frames_dir=kareler_dizini, cache_dir=ds_onbellegi
    )
    from src.data.dataset import match_grouped_split
    bolumler = match_grouped_split(ds, seed=42)
    test_ds = ds[bolumler["test"]]
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XPassGNN().to(device)
    model.load_state_dict(torch.load(kontrol_noktasi, weights_only=True, map_location=device))

    # GNN metrikleri.
    y_test, yhat_gnn = _gnn_tahmin(model, test_loader, device)
    gnn_metrikleri = {
        "mae": mae(y_test, yhat_gnn),
        "mse": mse(y_test, yhat_gnn),
        "spearman": spearman(y_test, yhat_gnn),
    }

    # Temel: xT'yi TÜM eylemlere göre hesapla, ardından test eylemleri üzerinde tahmin yap.
    tum_aksiyonlar = competition_actions(kareler_dizini)
    aksiyonlar_ps = tum_aksiyonlar[tum_aksiyonlar["type"].isin(("pass", "shot"))].reset_index(drop=True)
    xt = XTDeltaLabels(max_iter=50).fit(tum_aksiyonlar)
    temel = XTDeltaBaseline(xt_grid=xt.xt, g_grid=xt.g)
    test_satirlari = aksiyonlar_ps.iloc[bolumler["test"]].reset_index(drop=True)
    yhat_temel = temel.predict_from_actions(
        start_x=test_satirlari["start_x"].to_numpy(),
        start_y=test_satirlari["start_y"].to_numpy(),
        end_x=test_satirlari["end_x"].to_numpy(),
        end_y=test_satirlari["end_y"].to_numpy(),
        types=test_satirlari["type"].to_numpy(),
        success=test_satirlari["success"].to_numpy(dtype=bool),
    )
    temel_metrikleri = {
        "mae": mae(y_test, yhat_temel),
        "mse": mse(y_test, yhat_temel),
        "spearman": spearman(y_test, yhat_temel),
    }

    # ADR-004 birincil tripwire kontrolü.
    tripwire_atti = (gnn_metrikleri["spearman"] > 0.98) or (gnn_metrikleri["mae"] < 0.005)

    veri = {
        "label": "xt_delta",
        "n_test": int(len(y_test)),
        "gnn": gnn_metrikleri,
        "baseline_xt": temel_metrikleri,
        "delta_mae": gnn_metrikleri["mae"] - temel_metrikleri["mae"],
        "delta_spearman": gnn_metrikleri["spearman"] - temel_metrikleri["spearman"],
        "adr_004_birincil_tripwire_atti": tripwire_atti,
    }
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    cikti_yolu.write_text(json.dumps(veri, indent=2), encoding="utf-8")
    print(json.dumps(veri, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

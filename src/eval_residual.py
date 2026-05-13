"""Artık kafası GAT'ı xT temeline karşı değerlendirir.

Bir aksiyon için nihai DQS = `temel_tahmin + gat_artik_tahmini`. Sonuçta elde edilen
DQS'yi test bölümündeki orijinal `etiket` ile karşılaştırır.

`data/processed/sprint4_metrics.json` dosyasına şunları yazar:
  - artık ölçek metrikleri
  - nihai DQS metrikleri
  - ADR-005 tripwire bayrakları

Proje kökünden çalıştır:

    python -m src.eval_residual
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import XPassDataset, match_grouped_split
from src.models.xpass_gnn import XPassGNN
from src.utils.metrics import mae, mse, spearman

SPRINT2_TEMEL_TEST_MAE = 0.007900140521893728  # ADR-005 birincil eşik


def _gnn_tahmin(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Yükleyici üzerindeki (etiket_veya_artik_y, gnn_tahmini) döndürür."""
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
    kontrol_noktasi = proje_koku / "models" / "xpass_gnn" / "sprint4" / "best.pt"
    temel_yolu = proje_koku / "models" / "xpass_gnn" / "sprint4" / "baseline_preds.npy"
    gecmis_yolu = proje_koku / "models" / "xpass_gnn" / "sprint4" / "history.json"
    cikti_yolu = proje_koku / "data" / "processed" / "sprint4_metrics.json"

    ds = XPassDataset(
        labels_parquet=etiket_parquet, frames_dir=kareler_dizini, cache_dir=ds_onbellegi
    )
    bolumler = match_grouped_split(ds, seed=42)
    test_ds = ds[bolumler["test"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XPassGNN().to(device)
    model.load_state_dict(torch.load(kontrol_noktasi, weights_only=True, map_location=device))

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    y_test_ham, yhat_artik = _gnn_tahmin(model, test_loader, device)

    # Temel tahminleri aynı sırayla yükle.
    temel_hepsi = np.load(temel_yolu)
    test_temeli = temel_hepsi[bolumler["test"]]

    y_test = y_test_ham  # orijinal etiketler
    yhat_dqs = yhat_artik + test_temeli
    artik_y = y_test - test_temeli  # tanılama için artık ölçek

    artik_metrikleri = {
        "mae": mae(artik_y, yhat_artik),
        "mse": mse(artik_y, yhat_artik),
        "spearman": spearman(artik_y, yhat_artik),
    }
    dqs_metrikleri = {
        "mae": mae(y_test, yhat_dqs),
        "mse": mse(y_test, yhat_dqs),
        "spearman": spearman(y_test, yhat_dqs),
    }
    temel_metrikleri = {
        "mae": mae(y_test, test_temeli),
        "mse": mse(y_test, test_temeli),
        "spearman": spearman(y_test, test_temeli),
    }

    # ADR-005 tripwire'ları.
    birincil_atti = not (dqs_metrikleri["mae"] < SPRINT2_TEMEL_TEST_MAE)
    ucuncu_atti = False
    if gecmis_yolu.exists():
        gecmis = json.loads(gecmis_yolu.read_text(encoding="utf-8"))
        if gecmis.get("history"):
            v0 = gecmis["history"][0]["val_loss"]
            art5 = 0
            for r in gecmis["history"]:
                if r["val_loss"] > 2 * v0:
                    art5 += 1
                    if art5 >= 5:
                        ucuncu_atti = True
                        break
                else:
                    art5 = 0

    veri = {
        "label": "xt_delta",
        "n_test": int(len(y_test)),
        "sprint2_temel_test_mae": SPRINT2_TEMEL_TEST_MAE,
        "artik_olcek": artik_metrikleri,
        "final_dqs": dqs_metrikleri,
        "temel_metrikleri": temel_metrikleri,
        "temel_uzerinden_mae_iyilesme": temel_metrikleri["mae"] - dqs_metrikleri["mae"],
        "temel_uzerinden_spearman_iyilesme": dqs_metrikleri["spearman"] - temel_metrikleri["spearman"],
        "adr_005_birincil_tripwire_atti": birincil_atti,
        "adr_005_ucuncu_tripwire_atti": ucuncu_atti,
    }

    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    cikti_yolu.write_text(json.dumps(veri, indent=2), encoding="utf-8")
    print(json.dumps(veri, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

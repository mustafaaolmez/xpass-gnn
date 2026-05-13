"""Sprint-2 eğitilmiş GAT için Sprint-3 tanılamaları.

`sprint2/best.pt` ve önbelleğe alınmış `XPassDataset` yüklenir. Şunlar hesaplanır:

* GAT için egitim / dogrulama / test MAE ve Spearman ρ.
* Deterministik xT-temeli için aynı bölüm başına metrikler.
* Her iki model için yalnızca pas ve yalnızca şut metrikleri.
* GAT'ın **artık tahmin Spearman'ı**: ρ(y_gercek − y_temel, y_gat − y_temel).
  Pozitifse GAT kapalı-form temel ötesinde bilgi katmaktadır.

Her şeyi `data/processed/sprint3_diagnostics.json` dosyasına yazar ve insan
okunabilir özet yazdırır.

Proje kökünden çalıştır:

    python -m src.diagnostics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import XPassDataset, match_grouped_split
from src.data.events_to_actions import competition_actions
from src.data.labels.xt_delta import XTDeltaLabels
from src.models.baselines import XTDeltaBaseline
from src.models.xpass_gnn import XPassGNN
from src.utils.metrics import mae, mse, spearman

GNN_AKSIYON_TURLERI = ("pass", "shot")


def _gnn_tahmin(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Yükleyici üzerinde (y_gercek, y_tahmin) numpy dizileri döndürür."""
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


def _bolum_metrikleri(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    return {"mae": mae(y, yhat), "mse": mse(y, yhat), "spearman": spearman(y, yhat)}


def run(
    labels_parquet: Path,
    frames_dir: Path,
    ds_cache: Path,
    ckpt_path: Path,
    out_path: Path,
) -> dict:
    """Tam tanılama yükünü hesaplar. Döndürür ve JSON'a yazar."""
    ds = XPassDataset(
        labels_parquet=labels_parquet, frames_dir=frames_dir, cache_dir=ds_cache
    )
    bolumler = match_grouped_split(ds, seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XPassGNN().to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))

    # Bölüm başına GNN tahminleri.
    bolum_gnn: dict[str, dict[str, float]] = {}
    bolum_y: dict[str, np.ndarray] = {}
    bolum_yhat_gnn: dict[str, np.ndarray] = {}
    for bolum_adi in ("train", "val", "test"):
        alt = ds[bolumler[bolum_adi]]
        loader = DataLoader(alt, batch_size=256, shuffle=False)
        y, yhat = _gnn_tahmin(model, loader, device)
        bolum_y[bolum_adi] = y
        bolum_yhat_gnn[bolum_adi] = yhat
        bolum_gnn[bolum_adi] = _bolum_metrikleri(y, yhat)

    # Temel metrikleri.
    tum_aksiyonlar = competition_actions(frames_dir)
    xt = XTDeltaLabels(max_iter=50).fit(tum_aksiyonlar)
    temel = XTDeltaBaseline(xt_grid=xt.xt, g_grid=xt.g)
    aksiyonlar_ps = tum_aksiyonlar[
        tum_aksiyonlar["type"].isin(GNN_AKSIYON_TURLERI)
    ].reset_index(drop=True)

    bolum_temel: dict[str, dict[str, float]] = {}
    bolum_yhat_temel: dict[str, np.ndarray] = {}
    for bolum_adi in ("train", "val", "test"):
        satirlar = aksiyonlar_ps.iloc[bolumler[bolum_adi]].reset_index(drop=True)
        yhat_t = temel.predict_from_actions(
            start_x=satirlar["start_x"].to_numpy(),
            start_y=satirlar["start_y"].to_numpy(),
            end_x=satirlar["end_x"].to_numpy(),
            end_y=satirlar["end_y"].to_numpy(),
            types=satirlar["type"].to_numpy(),
            success=satirlar["success"].to_numpy(dtype=bool),
        )
        bolum_yhat_temel[bolum_adi] = yhat_t
        bolum_temel[bolum_adi] = _bolum_metrikleri(bolum_y[bolum_adi], yhat_t)

    # Test bölümünde yalnızca pas / yalnızca şut metrikleri.
    test_satirlari = aksiyonlar_ps.iloc[bolumler["test"]].reset_index(drop=True)
    tur_dizi = test_satirlari["type"].to_numpy()
    pas_mi = tur_dizi == "pass"
    sut_mu = tur_dizi == "shot"

    tur_bazi: dict[str, dict[str, dict[str, float]]] = {"pass": {}, "shot": {}}
    if pas_mi.any():
        tur_bazi["pass"]["gnn"] = _bolum_metrikleri(
            bolum_y["test"][pas_mi], bolum_yhat_gnn["test"][pas_mi]
        )
        tur_bazi["pass"]["baseline"] = _bolum_metrikleri(
            bolum_y["test"][pas_mi], bolum_yhat_temel["test"][pas_mi],
        )
    if sut_mu.any():
        tur_bazi["shot"]["gnn"] = _bolum_metrikleri(
            bolum_y["test"][sut_mu], bolum_yhat_gnn["test"][sut_mu]
        )
        tur_bazi["shot"]["baseline"] = _bolum_metrikleri(
            bolum_y["test"][sut_mu], bolum_yhat_temel["test"][sut_mu],
        )
    tur_bazi["pass"]["n"] = int(pas_mi.sum())
    tur_bazi["shot"]["n"] = int(sut_mu.sum())

    # Test üzerinde artık tahmin Spearman'ı.
    r_gercek = bolum_y["test"] - bolum_yhat_temel["test"]
    r_tahmin = bolum_yhat_gnn["test"] - bolum_yhat_temel["test"]
    artik_spearman = spearman(r_gercek, r_tahmin)
    artik_mae_val = mae(r_gercek, r_tahmin)

    # H1 (az uyum): egitim_spearman ≈ dogrulama_spearman ≈ test_spearman, hepsi düşük.
    h1_tutarli = (
        abs(bolum_gnn["train"]["spearman"] - bolum_gnn["test"]["spearman"]) < 0.15
    )
    h2_tutarli = (
        bolum_gnn["train"]["spearman"] - bolum_gnn["test"]["spearman"] > 0.15
    )

    veri = {
        "gnn_bolum_bazi": bolum_gnn,
        "temel_bolum_bazi": bolum_temel,
        "test_aksiyon_tur_bazi": tur_bazi,
        "artik": {
            "spearman": artik_spearman,
            "mae": artik_mae_val,
            "n": int(len(r_gercek)),
            "yorum": (
                "pozitif => GAT temelin otesinde bilgi ekliyor; "
                "~0 => GAT temel gurultusuyle eslesir; "
                "negatif => GAT sapmalari etiket sapmalarıyla ters iliskili."
            ),
        },
        "hipotez_kontrolu": {
            "h1_az_uyum_tutarli": h1_tutarli,
            "h2_uyumsuzluk_tutarli": h2_tutarli,
            "h1_h2_belirsiz": (not h1_tutarli) and (not h2_tutarli),
            "egitim_eksi_test_spearman": (
                bolum_gnn["train"]["spearman"] - bolum_gnn["test"]["spearman"]
            ),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(veri, indent=2), encoding="utf-8")
    return veri


def main() -> int:  # pragma: no cover - CLI
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    proje_koku = Path(__file__).resolve().parent.parent
    veri = run(
        labels_parquet=proje_koku / "data" / "processed" / "wc2022_xt_labels.parquet",
        frames_dir=proje_koku / "data" / "raw" / "comp43_season106",
        ds_cache=proje_koku / "data" / "processed" / "xpass_dataset" / "tam",
        ckpt_path=proje_koku / "models" / "xpass_gnn" / "sprint2" / "best.pt",
        out_path=proje_koku / "data" / "processed" / "sprint3_tanilama.json",
    )
    print(json.dumps(veri, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

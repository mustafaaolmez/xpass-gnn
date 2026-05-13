"""GAT'ı **artık** `etiket - temel(baslangic, bitis)` değerini tahmin etmek üzere eğitir.

ADR-005'e göre (Δ2 artık kafası):
  - Bir aksiyon için nihai DQS = `temel_tahmin + gat_artik_tahmini`.
  - Kapalı-form temel (uygun ızgara üzerinden xT-delta) toplamsal öncüldür;
    GAT yalnızca aramanın kaçırdığını öğrenir.

Strateji: `src/train.py::train()` uctan uca yeniden kullanılır — her grafın `y`
değerini eğitim döngüsüne vermeden önce bellekte `etiket - temel_tahmin` olarak
yeniden yazarız. Eğitilen kontrol noktası bu nedenle bir "artık" modeldir; değerlendirme
zamanında tahminin üstüne temel eklenir.

Çıktılar:
  - `models/xpass_gnn/sprint4/best.pt` + `history.json`
  - `models/xpass_gnn/sprint4/baseline_preds.npy` (önceden hesaplanmış öncel;
    şekil `(N,)` veri seti sırasına göre indekslenmiş — değerlendirme için gerekli).

Proje kökünden çalıştır:

    python -m src.train_residual
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import XPassDataset, match_grouped_split
from src.data.events_to_actions import competition_actions
from src.data.labels.xt_delta import XTDeltaLabels
from src.models.baselines import XTDeltaBaseline
from src.train import EgitimKonfigurasyonu, train

GNN_AKSIYON_TURLERI = ("pass", "shot")


def precompute_baseline_preds(frames_dir: Path, n_expected: int) -> np.ndarray:
    """WC2022 müsabakasındaki her pas+şut satırı için deterministik
    xT-temel tahminini hesaplar; `XPassDataset`'in kullandığı sırayla.
    """
    tum_aksiyonlar = competition_actions(frames_dir)
    xt = XTDeltaLabels(max_iter=50).fit(tum_aksiyonlar)
    temel = XTDeltaBaseline(xt_grid=xt.xt, g_grid=xt.g)
    aksiyonlar_ps = tum_aksiyonlar[
        tum_aksiyonlar["type"].isin(GNN_AKSIYON_TURLERI)
    ].reset_index(drop=True)
    tahminler = temel.predict_from_actions(
        start_x=aksiyonlar_ps["start_x"].to_numpy(),
        start_y=aksiyonlar_ps["start_y"].to_numpy(),
        end_x=aksiyonlar_ps["end_x"].to_numpy(),
        end_y=aksiyonlar_ps["end_y"].to_numpy(),
        types=aksiyonlar_ps["type"].to_numpy(),
        success=aksiyonlar_ps["success"].to_numpy(dtype=bool),
    )
    if len(tahminler) != n_expected:
        raise RuntimeError(
            f"Temel tahmin sayısı {len(tahminler)} != veri seti boyutu {n_expected}. "
            "Pas+şut filtre sıralaması değişmiş olabilir."
        )
    return tahminler.astype(np.float32)


def _rewrite_labels_to_residuals(
    dataset: XPassDataset, baseline_preds: np.ndarray
) -> None:
    """`dataset._data.y` değerini bellekte `etiket - temel_tahmin` olarak değiştirir."""
    y_duz = dataset._data.y
    if y_duz.dim() > 1:
        y_duz = y_duz.squeeze(-1)
    if y_duz.shape[0] != len(baseline_preds):
        raise RuntimeError(
            f"y_duz şekli {tuple(y_duz.shape)} != temel_tahminler şekli {baseline_preds.shape}"
        )
    yeni_y = y_duz - torch.tensor(baseline_preds, dtype=y_duz.dtype)
    if dataset._data.y.dim() > 1:
        dataset._data.y = yeni_y.unsqueeze(-1)
    else:
        dataset._data.y = yeni_y


def main() -> int:  # pragma: no cover - CLI
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    proje_koku = Path(__file__).resolve().parent.parent
    etiket_parquet = proje_koku / "data" / "processed" / "wc2022_xt_labels.parquet"
    kareler_dizini = proje_koku / "data" / "raw" / "comp43_season106"
    ds_onbellegi = proje_koku / "data" / "processed" / "xpass_dataset" / "tam"
    cikti_dizini = proje_koku / "models" / "xpass_gnn" / "sprint4"
    cikti_dizini.mkdir(parents=True, exist_ok=True)
    bolum_yolu = proje_koku / "data" / "processed" / "splits.json"
    temel_yolu = cikti_dizini / "baseline_preds.npy"

    print("Önbelleğe alınmış XPassDataset yükleniyor …")
    ds = XPassDataset(
        labels_parquet=etiket_parquet, frames_dir=kareler_dizini, cache_dir=ds_onbellegi
    )
    n = len(ds)
    print(f"  veri seti: {n} graf")

    print("Temel tahminler önceden hesaplanıyor (deterministik xT-delta) …")
    temel_tahminler = precompute_baseline_preds(kareler_dizini, n_expected=n)
    np.save(temel_yolu, temel_tahminler)
    print(f"  temel tahminler kaydedildi: {temel_yolu.relative_to(proje_koku)}")
    print(
        f"  temel tahmin istatistikleri: ortalama={temel_tahminler.mean():.5f}, "
        f"std={temel_tahminler.std():.5f}, min={temel_tahminler.min():.5f}, "
        f"maks={temel_tahminler.max():.5f}"
    )

    print("y → y − temel_tahmin (artık etiketler) yeniden yazılıyor …")
    _rewrite_labels_to_residuals(ds, temel_tahminler)
    artik = ds._data.y.squeeze(-1) if ds._data.y.dim() > 1 else ds._data.y
    print(
        f"  artık istatistikleri: ortalama={artik.mean().item():.5f}, "
        f"std={artik.std().item():.5f}, "
        f"min={artik.min().item():.5f}, maks={artik.max().item():.5f}"
    )

    print("Deterministik 80/10/10 bölüm oluşturuluyor (seed=42) …")
    bolumler = match_grouped_split(ds, seed=42, persist_path=bolum_yolu)
    print(
        f"  bölümler — egitim {len(bolumler['train'])}, "
        f"dogrulama {len(bolumler['val'])}, test {len(bolumler['test'])}"
    )

    kfg = EgitimKonfigurasyonu()  # Sprint 2 ile aynı hiperparametreler
    sonuc = train(ds, bolumler, kfg, out_dir=cikti_dizini)
    print("\nTEST METRİKLERİ (artık ölçek):")
    print(json.dumps(sonuc["test"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

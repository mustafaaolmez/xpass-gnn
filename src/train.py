"""XPassGNN eğitim döngüsü.

Mimari §3.3-§3.4'e göre: Smooth-L1 (Huber) regresyon, AdamW (lr=1e-3,
wd=1e-4), 1-epoch ısınmalı kosinüs LR, AMP karışık hassasiyet, parti 128.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader

from src.data.dataset import XPassDataset, match_grouped_split
from src.models.xpass_gnn import XPassGNN
from src.utils.metrics import mae, mse, spearman
from src.utils.seeding import seed_everything


@dataclass
class EgitimKonfigurasyonu:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    seed: int = 42
    hidden: int = 128
    heads: int = 4
    dropout: float = 0.1
    amp: bool = True


# Geriye dönük uyumluluk için takma ad
TrainConfig = EgitimKonfigurasyonu


def _kosinuslr(epoch: int, toplam: int, isinma: int, temel_lr: float) -> float:
    if epoch < isinma:
        return temel_lr * (epoch + 1) / max(1, isinma)
    ilerleme = (epoch - isinma) / max(1, toplam - isinma)
    return 0.5 * temel_lr * (1.0 + math.cos(math.pi * ilerleme))


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, amp: bool
) -> dict:
    model.eval()
    ys = []
    ytahminler = []
    toplam_kayip = 0.0
    n = 0
    with torch.no_grad():
        for parti in loader:
            parti = parti.to(device)
            with autocast(device_type=device.type, enabled=amp):
                tahmin = model(
                    parti.x, parti.edge_index, parti.edge_attr, parti.batch
                ).squeeze(-1)
                hedef = parti.y.squeeze(-1) if parti.y.dim() > 1 else parti.y
                kayip = F.smooth_l1_loss(tahmin, hedef, reduction="sum")
            toplam_kayip += float(kayip.item())
            n += hedef.numel()
            ys.append(hedef.detach().cpu().numpy())
            ytahminler.append(tahmin.detach().cpu().numpy())
    y = np.concatenate(ys)
    yh = np.concatenate(ytahminler)
    return {
        "loss": toplam_kayip / max(1, n),
        "mae": mae(y, yh),
        "mse": mse(y, yh),
        "spearman": spearman(y, yh),
        "n": n,
    }


def train(
    dataset: XPassDataset,
    splits: dict[str, list[int]],
    config: EgitimKonfigurasyonu,
    out_dir: Path,
) -> dict:
    """Modeli eğitir ve geçmişi + son metrikleri döndürür."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    egitim_ds = dataset[splits["train"]]
    dogrulama_ds = dataset[splits["val"]]
    test_ds = dataset[splits["test"]]

    egitim_loader = DataLoader(egitim_ds, batch_size=config.batch_size, shuffle=True)
    dogrulama_loader = DataLoader(dogrulama_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    model = XPassGNN(
        hidden=config.hidden, heads=config.heads, dropout=config.dropout
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = GradScaler(
        device=device.type, enabled=config.amp and device.type == "cuda"
    )

    gecmis: list[dict] = []
    en_iyi_dogrulama_mae = float("inf")
    en_iyi_epoch = -1
    t0 = time.time()
    for epoch in range(config.epochs):
        lr_simdi = _kosinuslr(epoch, config.epochs, config.warmup_epochs, config.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr_simdi

        model.train()
        epoch_kayip = 0.0
        n = 0
        for parti in egitim_loader:
            parti = parti.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(
                device_type=device.type, enabled=config.amp and device.type == "cuda"
            ):
                tahmin = model(
                    parti.x, parti.edge_index, parti.edge_attr, parti.batch
                ).squeeze(-1)
                hedef = parti.y.squeeze(-1) if parti.y.dim() > 1 else parti.y
                kayip = F.smooth_l1_loss(tahmin, hedef)
            scaler.scale(kayip).backward()
            scaler.step(opt)
            scaler.update()
            epoch_kayip += float(kayip.item()) * hedef.numel()
            n += hedef.numel()

        egitim_kayip = epoch_kayip / max(1, n)
        dogrulama_metrikleri = evaluate(model, dogrulama_loader, device, config.amp)
        gecen = time.time() - t0
        kayit = {
            "epoch": epoch,
            "lr": lr_simdi,
            "train_loss": egitim_kayip,
            "val_loss": dogrulama_metrikleri["loss"],
            "val_mae": dogrulama_metrikleri["mae"],
            "val_mse": dogrulama_metrikleri["mse"],
            "val_spearman": dogrulama_metrikleri["spearman"],
            "elapsed_s": gecen,
        }
        gecmis.append(kayit)
        print(
            f"epoch {epoch:3d}  egitim_kayip {egitim_kayip:.5f}  dogrulama_kayip {dogrulama_metrikleri['loss']:.5f}  "
            f"dogrulama_mae {dogrulama_metrikleri['mae']:.5f}  dogrulama_rho {dogrulama_metrikleri['spearman']:.4f}  "
            f"({gecen:.1f}s)"
        )

        if dogrulama_metrikleri["mae"] < en_iyi_dogrulama_mae:
            en_iyi_dogrulama_mae = dogrulama_metrikleri["mae"]
            en_iyi_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best.pt")

    # En iyi kontrol noktasıyla son test değerlendirmesi.
    model.load_state_dict(torch.load(out_dir / "best.pt", weights_only=True))
    test_metrikleri = evaluate(model, test_loader, device, config.amp)

    gecmis_yolu = out_dir / "history.json"
    gecmis_yolu.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "history": gecmis,
                "best_epoch": en_iyi_epoch,
                "test": test_metrikleri,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"history": gecmis, "test": test_metrikleri, "best_epoch": en_iyi_epoch}


def main() -> int:  # pragma: no cover - CLI girişi
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    proje_koku = Path(__file__).resolve().parent.parent
    etiket_parquet = proje_koku / "data" / "processed" / "wc2022_xt_labels.parquet"
    kareler_dizini = proje_koku / "data" / "raw" / "comp43_season106"
    ds_onbellegi = proje_koku / "data" / "processed" / "xpass_dataset" / "tam"
    cikti_dizini = proje_koku / "models" / "xpass_gnn" / "sprint2"
    bolum_yolu = proje_koku / "data" / "processed" / "splits.json"

    print("Tam veri seti yükleniyor (ilk seferde birkaç dakika sürebilir) …")
    ds = XPassDataset(
        labels_parquet=etiket_parquet, frames_dir=kareler_dizini, cache_dir=ds_onbellegi
    )
    print(f"  veri seti: {len(ds)} graf")

    bolumler = match_grouped_split(ds, seed=42, persist_path=bolum_yolu)
    print(
        f"  bölümler — egitim {len(bolumler['train'])}, dogrulama {len(bolumler['val'])}, test {len(bolumler['test'])}"
    )

    sonuc = train(ds, bolumler, EgitimKonfigurasyonu(), out_dir=cikti_dizini)
    print("\nTEST METRİKLERİ:")
    print(json.dumps(sonuc["test"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

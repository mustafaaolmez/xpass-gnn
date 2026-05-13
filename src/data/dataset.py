"""PyG veri seti: etiketli WC2022 aksiyon başına graflar.

`data/processed/<komp>_xt_labels.parquet` ve 360 karelerini
`data/raw/<komp>/<mac_id>_frames.json` dosyasından okur, grafları oluşturur ve
`data/processed/<komp>/dataset.pt` dosyasına önbelleğe alır.

Yalnızca pas + şut aksiyonları denetimli öğrenmeye tabi tutulur (proje hedefine göre).
Taşıma aksiyonları veri seti oluşturma sırasında filtrelenir.

Her `Data` nesnesi, akış kodunun parquet'i yeniden okumadan maç bazlı
eğitim/doğrulama/test bölümü oluşturabilmesi için `match_id` (int tensör) depolar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset

from src.data.graph_builder import build_graph
from src.data.statsbomb_loader import load_competition_frames

GNN_AKSIYON_TURLERI = ("pass", "shot")


class XPassDataset(InMemoryDataset):
    """Her pas veya şut aksiyonu için bir graf; denetimli hedef = seçilen etiket."""

    def __init__(
        self,
        labels_parquet: Path,
        frames_dir: Path,
        cache_dir: Path,
        label_column: str = "xt_delta",
        max_actions: int | None = None,
    ) -> None:
        self.labels_parquet = Path(labels_parquet)
        self.frames_dir = Path(frames_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.label_column = label_column
        self.max_actions = max_actions
        super().__init__(root=str(self.cache_dir), transform=None, pre_transform=None)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:  # pragma: no cover - PyG altyapısı
        return []

    @property
    def processed_file_names(self) -> list[str]:
        sonek = f"_max{self.max_actions}" if self.max_actions else "_tam"
        return [f"xpass_{self.label_column}{sonek}.pt"]

    def download(self) -> None:  # pragma: no cover - veri zaten yerel
        pass

    def process(self) -> None:
        aksiyonlar = pd.read_parquet(self.labels_parquet)
        aksiyonlar = aksiyonlar[aksiyonlar["type"].isin(GNN_AKSIYON_TURLERI)].reset_index(
            drop=True
        )
        if self.max_actions is not None:
            aksiyonlar = aksiyonlar.head(self.max_actions)
        etiketler = aksiyonlar[self.label_column].to_numpy()
        mac_idleri = aksiyonlar["match_id"].to_numpy()

        kareler = load_competition_frames(self.frames_dir)

        veri_listesi = []
        for i, satir in aksiyonlar.iterrows():
            ff = kareler.get(satir["action_id"])
            veri = build_graph(
                action={
                    "start_x": satir["start_x"],
                    "start_y": satir["start_y"],
                    "end_x": satir["end_x"],
                    "end_y": satir["end_y"],
                    "type": satir["type"],
                    "success": bool(satir["success"]),
                },
                freeze_frame=ff,
                label=float(etiketler[i]),
            )
            veri.match_id = torch.tensor([int(mac_idleri[i])], dtype=torch.long)
            veri_listesi.append(veri)
        self.save(veri_listesi, self.processed_paths[0])


def match_grouped_split(
    dataset: XPassDataset,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    persist_path: Path | None = None,
) -> dict[str, list[int]]:
    """Maç bazında bölünmüş {"train": [idx...], "val": [...], "test": [...]} döndürür.
    Hiçbir maç iki bölüme katkıda bulunmaz.

    `seed` ile tekrar üretilebilir. `persist_path` ayarlanırsa sonuçta elde edilen
    {mac_id: bolum} eşlemesini denetlenebilirlik için JSON olarak yazar.
    """
    # Graf başına match_id'leri topla.
    graf_mac = np.asarray([int(d.match_id.item()) for d in dataset])
    benzersiz_maclar = np.array(sorted(np.unique(graf_mac)))
    rng = np.random.default_rng(seed)
    rng.shuffle(benzersiz_maclar)
    n = len(benzersiz_maclar)
    n_egitim = int(n * train_frac)
    n_dogrulama = int(n * val_frac)
    egitim_maclari = set(benzersiz_maclar[:n_egitim].tolist())
    dogrulama_maclari = set(benzersiz_maclar[n_egitim : n_egitim + n_dogrulama].tolist())
    # Kalanlar teste gider.
    bolumler: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, m in enumerate(graf_mac):
        if m in egitim_maclari:
            bolumler["train"].append(i)
        elif m in dogrulama_maclari:
            bolumler["val"].append(i)
        else:
            bolumler["test"].append(i)

    if persist_path is not None:
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        mac_bolum = (
            {int(m): "train" for m in egitim_maclari}
            | {int(m): "val" for m in dogrulama_maclari}
            | {int(m): "test" for m in benzersiz_maclar[n_egitim + n_dogrulama :].tolist()}
        )
        persist_path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "train_frac": train_frac,
                    "val_frac": val_frac,
                    "n_egitim_mac": len(egitim_maclari),
                    "n_dogrulama_mac": len(dogrulama_maclari),
                    "n_test_mac": int(n - n_egitim - n_dogrulama),
                    "n_egitim_graf": len(bolumler["train"]),
                    "n_dogrulama_graf": len(bolumler["val"]),
                    "n_test_graf": len(bolumler["test"]),
                    "mac_bolum": mac_bolum,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return bolumler

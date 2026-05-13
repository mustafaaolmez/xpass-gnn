"""Aksiyon başına graf oluşturucu.

Her aksiyon bir PyG `Data` nesnesine dönüştürülür:

  Düğüm özellikleri (12 boyut):
    [x_norm, y_norm, takım_arkadası_mı, oyuncu_mu, kaleci_mi, pozisyon_onehot[4], top_mu, vx, vy]

  Kenar özellikleri (4 boyut):
    [mesafe_norm, dx, dy, ayni_takim_bayragi]

  Kenarlar: mevcut tüm düğümler arasında tam bağlantılı (her iki yönde).

  360 karesi olmayan olaylar için yedek: 3 düğümlü iskelet
    (oyuncu, top, "bitiş_konumundaki_hedef") + `has_360` grafik düzeyi skalar niteliği
    (360 karesi varsa 1.0, yoksa 0.0).

Hızlar sıfır tutulmuştur; StatsBomb 360 kareleri oyuncu başına hız içermez.
`vx/vy` sütunları gelecekteki zenginleştirme için ayrılmıştır.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from src.utils.pitch import PITCH_LEN, PITCH_WID

DUGUM_OZELLIK_BOYUTU = 12
KENAR_OZELLIK_BOYUTU = 4


def _oyuncu_dugumu(
    x: float, y: float, takım_arkadası: bool, oyuncu: bool, kaleci: bool
) -> list[float]:
    """Bir oyuncu düğümü özellik vektörü oluşturur.
    Pozisyon onehot 4 yuvadan oluşur:
    [KL, defans, orta saha, forvet] — şu an yalnızca StatsBomb 360 `keeper`
    bayrağından KL bilindiğinden, KL olmayanlar 3. yuvaya (forvet grubu) düşer.
    """
    # 4 yuvalu pozisyon onehot — KL kesin, geri kalanlar geçici grup.
    pos = [1.0, 0.0, 0.0, 0.0] if kaleci else [0.0, 0.0, 0.0, 1.0]
    return [
        float(x) / PITCH_LEN,
        float(y) / PITCH_WID,
        1.0 if takım_arkadası else 0.0,
        1.0 if oyuncu else 0.0,
        1.0 if kaleci else 0.0,
        *pos,
        0.0,  # top_mu
        0.0,  # vx
        0.0,  # vy
    ]


def _top_dugumu(x: float, y: float) -> list[float]:
    return [
        float(x) / PITCH_LEN,
        float(y) / PITCH_WID,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,  # top_mu
        0.0,
        0.0,
    ]


def _tam_baglantili(n: int) -> torch.Tensor:
    """Yönlü tam bağlantılı kenar indeksi (2, n*(n-1)) döndürür (öz-döngü yok)."""
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    src, dst = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mask = src != dst
    return torch.from_numpy(np.stack([src[mask], dst[mask]], axis=0).astype(np.int64))


def build_graph(
    action: dict,
    freeze_frame: list[dict] | None,
    label: float,
) -> Data:
    """Tek bir aksiyon için PyG Data nesnesi oluşturur.

    Parametreler:
        action: start_x, start_y, end_x, end_y, type, success içeren sözlük.
        freeze_frame: oyuncu sözlüklerinin listesi veya None/boş liste
            → 3 düğümlü iskelet yedek kullanılır.
        label: denetimli hedef.
    """
    var_360 = bool(freeze_frame)
    dugumler: list[list[float]] = []

    # Aksiyonun başlangıç konumundaki top (0. düğüm).
    dugumler.append(_top_dugumu(action["start_x"], action["start_y"]))

    if var_360:
        for p in freeze_frame:
            if p["x"] is None or p["y"] is None:
                continue
            dugumler.append(
                _oyuncu_dugumu(p["x"], p["y"], p["teammate"], p["actor"], p["keeper"])
            )
    else:
        # İskelet: başlangıçta sentetik "oyuncu", bitişte sentetik "hedef".
        dugumler.append(
            _oyuncu_dugumu(
                action["start_x"],
                action["start_y"],
                takım_arkadası=True,
                oyuncu=True,
                kaleci=False,
            )
        )
        dugumler.append(
            _oyuncu_dugumu(
                action["end_x"],
                action["end_y"],
                takım_arkadası=True,
                oyuncu=False,
                kaleci=False,
            )
        )

    x = torch.tensor(dugumler, dtype=torch.float32)

    # Kenar indeksi + özellikler.
    n = x.shape[0]
    edge_index = _tam_baglantili(n)
    if edge_index.numel():
        src = edge_index[0]
        dst = edge_index[1]
        dx = x[dst, 0] - x[src, 0]
        dy = x[dst, 1] - x[src, 1]
        dist = torch.sqrt(dx**2 + dy**2)
        # ayni_takim: top↔oyuncu kenarları her zaman 0; oyuncu↔oyuncu takım arkadaşı bayrağıyla eşleşir.
        src_oyuncu_mu = x[src, 9] < 0.5  # 9. sütundaki is_ball bayrağı
        dst_oyuncu_mu = x[dst, 9] < 0.5
        src_takim = x[src, 2]
        dst_takim = x[dst, 2]
        ayni_takim = (
            src_oyuncu_mu.float()
            * dst_oyuncu_mu.float()
            * (src_takim == dst_takim).float()
        )
        edge_attr = torch.stack([dist, dx, dy, ayni_takim], dim=1)
    else:
        edge_attr = torch.zeros((0, KENAR_OZELLIK_BOYUTU), dtype=torch.float32)

    y = torch.tensor([float(label)], dtype=torch.float32)
    veri = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    veri.has_360 = torch.tensor([1.0 if var_360 else 0.0], dtype=torch.float32)
    return veri

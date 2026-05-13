"""GNN'in `(B, 1)` regresyon arayüzünü paylaşan temel regresörler.

Bunlar "GNN olmayan bir model ne tahmin eder?" referans noktalarıdır.

xT-delta geçişi: denetimli etiket zaten xT-delta'dır (ADR-004'e göre),
bu nedenle geçiş modeli ızgara koordinatlarından xT(bitiş) − xT(başlangıç)
hesaplar. GNN'in bu temeli geçmesi gerekir.

Temel model dondurulmuş kare oyuncularını KULLANMAZ — yalnızca topun
başlangıç/bitiş koordinatlarını ve önceden hesaplanmış xT ızgarasını kullanır.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.data.labels.xt_delta import _zone_of  # vektörleştirilmiş (x,y)→(zx,zy)


class XTDeltaBaseline(nn.Module):
    """Graf verilerinden aksiyon başına xT-delta etiketini yeniden üretir.

    Kurucu, uygun xT ızgarasını (ve şutlar için g) numpy dizileri olarak kabul eder;
    bunları eğitilemez tamponlara dönüştürür. İleri geçiş, 0. düğümden
    (is_ball=1 olan top) top başlangıç/bitiş koordinatlarını çeker.

    Değerlendirici Sprint-2 karşılaştırması için :meth:`forward` değil,
    :meth:`predict_from_actions` kullanır.
    """

    def __init__(self, xt_grid: np.ndarray, g_grid: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("xt_grid", torch.tensor(xt_grid, dtype=torch.float32))
        self.register_buffer("g_grid", torch.tensor(g_grid, dtype=torch.float32))
        self.n_x, self.n_y = xt_grid.shape

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,  # noqa: ARG002 - GNN imzasıyla eşleşmek için mevcut
        edge_attr: torch.Tensor,  # noqa: ARG002
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Gruptaki her graf için top başlangıç konumundaki xT'yi döndürür."""
        n_graflar = int(batch.max().item()) + 1 if batch.numel() else x.shape[0]
        cihaz = x.device
        cikti = torch.zeros((n_graflar, 1), device=cihaz)
        for g in range(n_graflar):
            maske = batch == g
            graf_x = x[maske]
            top_satirlari = (graf_x[:, 9] > 0.5).nonzero(as_tuple=False)
            if top_satirlari.numel() == 0:
                continue
            top = graf_x[top_satirlari[0, 0]]
            zx = int(np.clip(np.floor(top[0].item() * self.n_x), 0, self.n_x - 1))
            zy = int(np.clip(np.floor(top[1].item() * self.n_y), 0, self.n_y - 1))
            cikti[g, 0] = self.xt_grid[zx, zy]
        return cikti

    def predict_from_actions(
        self,
        start_x: np.ndarray,
        start_y: np.ndarray,
        end_x: np.ndarray,
        end_y: np.ndarray,
        types: np.ndarray,
        success: np.ndarray,
    ) -> np.ndarray:
        """Her aksiyon için tam xT-delta etiketini hesaplar.

        `XTDeltaLabels.compute` ile aynı formülü uygular, ancak zaten uygun
        ızgarayı kullanarak. Sprint-2 değerlendirmesinde GNN tahminini
        başlangıç/bitiş koordinatlarına erişimi olan deterministik bir geçişle
        karşılaştırmak için kullanılır.
        """
        xt = self.xt_grid.cpu().numpy()
        g = self.g_grid.cpu().numpy()
        zsx, zsy = _zone_of(start_x, start_y, self.n_x, self.n_y)
        zex, zey = _zone_of(end_x, end_y, self.n_x, self.n_y)
        baslangic_xt = xt[zsx, zsy]
        bitis_xt = xt[zex, zey]
        sut_g = g[zsx, zsy]
        cikti = np.zeros(len(types))
        hareket_mi = (types == "pass") | (types == "carry")
        cikti = np.where(hareket_mi & success, bitis_xt - baslangic_xt, cikti)
        cikti = np.where(hareket_mi & ~success, -baslangic_xt, cikti)
        cikti = np.where(types == "shot", sut_g - baslangic_xt, cikti)
        return cikti

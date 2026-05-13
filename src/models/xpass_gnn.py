"""xPass++ GAT modeli.

Mimari (docs/architecture.md §3.2):
  3 katmanlı GAT, gizli katmanlarda 4 dikkat kafası, tek kafali çıktı projeksiyonu,
  gizli boyut 128 (kafa başı 32 × 4 kafa = 128 efektif), global ortalama+maksimum havuzlama,
  2 katmanlı MLP kafası → skaler regresyon.

Kenar nitelikleri (4 boyutlu), desteklenen GATConv-kenar varyantı tarafından kullanılır.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

DUGUM_GIRIS_BOYUTU = 12  # src.data.graph_builder.NODE_FEATURE_DIM ile eşleşir
KENAR_GIRIS_BOYUTU = 4


class XPassGNN(nn.Module):
    """3 katmanlı GAT kodlayıcı + regresyon kafası."""

    def __init__(
        self,
        node_dim_in: int = DUGUM_GIRIS_BOYUTU,
        edge_dim_in: int = KENAR_GIRIS_BOYUTU,
        hidden: int = 128,
        heads: int = 4,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        per_head = hidden // heads
        self.gat1 = GATConv(
            node_dim_in, per_head, heads=heads, edge_dim=edge_dim_in, dropout=dropout
        )
        self.gat2 = GATConv(
            hidden, per_head, heads=heads, edge_dim=edge_dim_in, dropout=dropout
        )
        # Son GAT katmanı kafalar genelinde ortalaması alır (concat=False).
        self.gat3 = GATConv(
            hidden,
            hidden,
            heads=heads,
            edge_dim=edge_dim_in,
            dropout=dropout,
            concat=False,
        )

        # Havuzlama = birleştir(ortalama, maksimum) → 2 × gizli boyut.
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr))
        h = F.elu(self.gat2(h, edge_index, edge_attr=edge_attr))
        h = F.elu(self.gat3(h, edge_index, edge_attr=edge_attr))
        pooled = torch.cat(
            [global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1
        )
        return self.head(pooled)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

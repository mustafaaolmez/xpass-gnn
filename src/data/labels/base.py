"""EtiketUreticisi protokolü — etiket seçiminin bağlandığı tek dikiş noktası.

Her GNN için denetimli hedef bu protokol üzerinden geçer. Şu anki uygulama
`XTDeltaLabels`tır (Singh 2018 xT). VAEP tabanlı üretici ADR-004 kapısıyla
beklemede tutulmaktadır ve kolayca takılabilir ikinci bir uygulama olacaktır.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class EtiketUreticisi(Protocol):
    """Aksiyon başına skaler denetimli hedef üretir.

    Uygulamalar parquet sütun adı olarak ve grafik başlıklarında kullanılan
    `.name` niteliğine sahiptir. `compute` girdi `actions` DataFrame'iyle
    aynı indekste bir Seri döndürür. Belirli bir aksiyon türü için etiket
    tanımsızsa NaN kabul edilir — akış veri seti bu satırları filtreler.
    """

    name: str

    def compute(
        self, actions: pd.DataFrame
    ) -> pd.Series:  # pragma: no cover - Protocol
        """``actions.index`` ile hizalanmış satır başına skaler etiket döndürür."""
        ...

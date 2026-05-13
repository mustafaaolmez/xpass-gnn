"""FIFA WC 2022 için xT-delta etiketli aksiyon parquet'i üretir.

`data/raw/comp43_season106/` dizinindeki StatsBomb olay JSON'larını okur,
`events_to_actions` → `XTDeltaLabels.fit().compute()` çalıştırır ve
`data/processed/wc2022_xt_labels.parquet` dosyasına yazar.

Proje kökünden çalıştır:

    python scripts/produce_labels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.data.events_to_actions import competition_actions
from src.data.labels.xt_delta import XTDeltaLabels

PROJE_KOKU = Path(__file__).resolve().parent.parent
KARELER_DIZINI = PROJE_KOKU / "data" / "raw" / "comp43_season106"
CIKTI_YOLU = PROJE_KOKU / "data" / "processed" / "wc2022_xt_labels.parquet"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"{KARELER_DIZINI.relative_to(PROJE_KOKU)} dizininden aksiyonlar yükleniyor …")
    aksiyonlar = competition_actions(KARELER_DIZINI)
    print(f"  {aksiyonlar['match_id'].nunique()} maçta {len(aksiyonlar)} aksiyon.")

    print("xT ızgarası hesaplanıyor + aksiyon başına xT-delta etiketleri üretiliyor …")
    xt = XTDeltaLabels(max_iter=50).fit(aksiyonlar)
    aksiyonlar["xt_delta"] = xt.compute(aksiyonlar)
    print(
        f"  xT ızgara aralığı: [{xt.xt.min():.3f}, {xt.xt.max():.3f}]; "
        f"yakınsama iterasyonu: {xt.iterations}; artık: {xt.residual:.2e}"
    )
    print(
        f"  etiket istatistikleri: ortalama={aksiyonlar['xt_delta'].mean():.5f}, "
        f"std={aksiyonlar['xt_delta'].std():.5f}, "
        f"min={aksiyonlar['xt_delta'].min():.5f}, "
        f"maks={aksiyonlar['xt_delta'].max():.5f}"
    )

    CIKTI_YOLU.parent.mkdir(parents=True, exist_ok=True)
    aksiyonlar.to_parquet(CIKTI_YOLU, engine="fastparquet", index=False)
    print(f"{CIKTI_YOLU.relative_to(PROJE_KOKU)} dosyasına yazıldı ({len(aksiyonlar)} satır).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

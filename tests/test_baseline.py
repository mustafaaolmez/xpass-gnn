"""xT-delta temeli için sağlamlık kontrolleri."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.events_to_actions import competition_actions
from src.data.labels.xt_delta import XTDeltaLabels
from src.models.baselines import XTDeltaBaseline

PROJE_KOKU = Path(__file__).resolve().parent.parent
WC22_DIZIN = PROJE_KOKU / "data" / "raw" / "comp43_season106"


def test_temel_xt_delta_etiketlerini_kurtarir() -> None:
    """Temel, etiketlerin hesaplandığı aynı xT ızgarasını kullanıyorsa,
    predict_from_actions çıktısı etiket üreticisinin compute çıktısıyla
    TAM olarak eşleşmelidir (kayan nokta hassasiyeti hariç).
    """
    aksiyonlar = competition_actions(WC22_DIZIN)
    xt = XTDeltaLabels(max_iter=300).fit(aksiyonlar)
    etiketler = xt.compute(aksiyonlar).to_numpy()

    temel = XTDeltaBaseline(xt_grid=xt.xt, g_grid=xt.g)
    yhat = temel.predict_from_actions(
        start_x=aksiyonlar["start_x"].to_numpy(),
        start_y=aksiyonlar["start_y"].to_numpy(),
        end_x=aksiyonlar["end_x"].to_numpy(),
        end_y=aksiyonlar["end_y"].to_numpy(),
        types=aksiyonlar["type"].to_numpy(),
        success=aksiyonlar["success"].to_numpy(dtype=bool),
    )

    # Temel, etiket formülüdür. Farklar sıfır (veya yakın) olmalıdır.
    assert np.allclose(yhat, etiketler, atol=1e-5), (
        f"temel, etiket üreticisinden farklı; maks mutlak fark "
        f"{float(np.max(np.abs(yhat - etiketler))):.6e}"
    )

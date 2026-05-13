"""xT-delta etiket üreticisi (Singh 2018) için birim testler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.events_to_actions import competition_actions
from src.data.labels.xt_delta import DEFAULT_NX, DEFAULT_NY, XTDeltaLabels

PROJE_KOKU = Path(__file__).resolve().parent.parent
WC22_DIZIN = PROJE_KOKU / "data" / "raw" / "comp43_season106"


def _sentetik_aksiyonlar() -> pd.DataFrame:
    """Net saldırı yönüne sahip sentetik ~1000 aksiyon corpus'u.

    Gerçek veri olmadan yakınsama ve şekil hatalarını yakalamaya yardımcı olur.
    Paslar çoğunlukla ileri gider (düşük x → yüksek x); birkaç şut hedefe yakından.
    """
    rng = np.random.default_rng(0)
    n = 1000
    bx = rng.uniform(0, 100, n)
    by = rng.uniform(0, 80, n)
    ex = np.clip(bx + rng.normal(15, 8, n), 0, 120)
    ey = np.clip(by + rng.normal(0, 6, n), 0, 80)
    basarili = rng.random(n) < 0.85
    df = pd.DataFrame(
        {
            "type": "pass",
            "start_x": bx,
            "start_y": by,
            "end_x": ex,
            "end_y": ey,
            "success": basarili,
        }
    )
    # Hede yakın 50 şut ekle.
    sut_x = rng.uniform(100, 115, 50)
    sut_y = rng.uniform(25, 55, 50)
    sutlar = pd.DataFrame(
        {
            "type": "shot",
            "start_x": sut_x,
            "start_y": sut_y,
            "end_x": np.full(50, 120.0),
            "end_y": sut_y,
            "success": rng.random(50) < 0.12,
        }
    )
    return pd.concat([df, sutlar], ignore_index=True)


def test_sentetikte_yakinsama() -> None:
    """Yalnızca yakınsama + şekil — sentetik yoğunluk istikrarlı uzaysal
    gradyan için çok düşük. Uzaysal yapı aşağıdaki gerçek WC2022 verisinde test edilir.
    """
    xt = XTDeltaLabels(max_iter=200, tol=1e-6)
    xt.fit(_sentetik_aksiyonlar())
    assert xt.xt is not None
    assert xt.xt.shape == (DEFAULT_NX, DEFAULT_NY)
    assert xt.iterations < xt.max_iter, "max_iter öncesinde yakınsama olmadı"
    assert xt.residual < 1e-6, f"artık {xt.residual}"
    # Tüm değerler negatif olmamalı (xT bir gol olasılığıdır).
    assert (xt.xt >= -1e-9).all()


def test_hesapla_sekil_ve_basarisiz_isaret() -> None:
    """Başarısız hareketlerin etiketi tanım gereği negatif olmamalıdır; şekil + NaN yok."""
    aksiyonlar = _sentetik_aksiyonlar()
    xt = XTDeltaLabels(max_iter=200).fit(aksiyonlar)
    etiketler = xt.compute(aksiyonlar)
    assert isinstance(etiketler, pd.Series)
    assert len(etiketler) == len(aksiyonlar)
    assert etiketler.name == "xt_delta"
    assert not etiketler.isna().any()
    # Başarısız hareketler: etiket = −xT(başlangıç) ≤ 0.
    basarisizlar = aksiyonlar[(aksiyonlar["type"] == "pass") & ~aksiyonlar["success"]]
    assert (etiketler.loc[basarisizlar.index] <= 1e-9).all()


def test_gercek_wc22_smoke() -> None:
    """Önbelleğe alınmış WC2022 maçlarında fit et; şekil, aralık ve
    xT'nin saldırı yarısında daha yüksek olduğunu doğrula (Singh formülasyonunun
    temel özelliği).
    """
    aksiyonlar = competition_actions(WC22_DIZIN)
    xt = XTDeltaLabels(max_iter=300).fit(aksiyonlar)
    assert xt.xt is not None
    # Gerçek xT genellikle [0, ~0.4] aralığındadır.
    assert xt.xt.min() >= -1e-9
    assert xt.xt.max() <= 0.6, f"xT max şüpheli yüksek: {xt.xt.max():.4f}"
    # Saldırı yarısı ortalaması > savunma yarısı ortalaması. Bu xT'nin temel özelliğidir.
    saldiri_yarisi = xt.xt[DEFAULT_NX // 2 :, :].mean()
    savunma_yarisi = xt.xt[: DEFAULT_NX // 2, :].mean()
    assert saldiri_yarisi > savunma_yarisi, (
        f"saldırı yarısı xT ({saldiri_yarisi:.4f}) savunma yarısını ({savunma_yarisi:.4f}) "
        f"geçmeli. Muhtemelen işaret veya bölge yönelimi hatası."
    )
    etiketler = xt.compute(aksiyonlar)
    assert not etiketler.isna().any()

"""Sprint-1 Phase-0 smoke: olaylar → SPADL-lite dönüştürücü önbelleğe alınmış JSON üzerinde çalışır."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.events_to_actions import (
    competition_actions,
    events_file_to_actions,
)

PROJE_KOKU = Path(__file__).resolve().parent.parent
WC22_DIZIN = PROJE_KOKU / "data" / "raw" / "comp43_season106"
ORNEK_OLAYLAR = WC22_DIZIN / "3857256_events.json"


def test_tek_mac_donusumu() -> None:
    """Önbelleğe alınmış bir olay dosyası boş olmayan aksiyon çerçevesine dönüşür."""
    assert ORNEK_OLAYLAR.exists(), f"eksik fixture: {ORNEK_OLAYLAR}"
    df = events_file_to_actions(ORNEK_OLAYLAR, match_id=3857256)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 100, f"100'den fazla aksiyon bekleniyordu; {len(df)} bulundu"
    # Şema kontrolü — her sütun mevcut olmalıdır.
    for sutun in (
        "match_id",
        "action_id",
        "period",
        "time_seconds",
        "team_name",
        "player_name",
        "type",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "success",
    ):
        assert sutun in df.columns, f"eksik sütun {sutun!r}"
    # Koordinatlar StatsBomb 120×80 saha sınırları içinde (şutlar için küçük tolerans).
    assert df["start_x"].between(0, 121).all()
    assert df["start_y"].between(0, 81).all()
    # Tipik bir maçta tüm hedef türleri mevcut.
    turler = set(df["type"].unique())
    assert {"pass", "shot"}.issubset(turler), f"beklenmedik türler: {turler}"


def test_musabaka_taramasi() -> None:
    """`competition_actions` tüm maç dosyalarını toplar."""
    df = competition_actions(WC22_DIZIN)
    assert len(df) > 300, f"3 maçta 300'den fazla aksiyon bekleniyordu; {len(df)} bulundu"
    # Birden fazla maç ID'si mevcut.
    assert df["match_id"].nunique() >= 1
    # Temel geometrik sütunlarda NaN yok.
    for sutun in ("start_x", "start_y", "end_x", "end_y"):
        assert not df[sutun].isna().any(), f"{sutun} NaN içeriyor"

"""Minimal StatsBomb olayları → SPADL-lite aksiyon çerçevesi.

Bu modül, önbelleğe alınmış statsbombpy JSON olay dosyalarını okur ve
akış etiket üreticilerinin ve graf oluşturucunun ihtiyaç duyduğu alanlara
sahip düzgün bir aksiyon çerçevesi döndürür:

    mac_id, aksiyon_id, periyot, sure_saniye, takim_id, takim_adi, oyuncu_id,
    oyuncu_adi, tur, baslangic_x, baslangic_y, bitis_x, bitis_y, basarili

Koordinatlar StatsBomb'un 120 × 80 saha uzayındadır; bu uzayda bırakılır
(normalleştirme graf oluşturucuda gerçekleşir, bkz. mimari §2.3).

Referanslar:
- StatsBomb olay alanı semantiği — https://github.com/statsbomb/open-data
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# xPass++ için önemli aksiyon türleri. "Taşıma" dahildir çünkü
# xT-delta top ilerleme aksiyonları için de tanımlanmıştır.
AKSIYON_TURLERI = ("Pass", "Shot", "Carry")


def _zaman_damgasi_saniyeye(ts: str) -> float:
    """StatsBomb "HH:MM:SS.mmm" → float saniye dönüşümü.

    StatsBomb zaman damgaları her periyotta sıfırlanır; periyot ayrı gösterilir.
    """
    parcalar = ts.split(":")
    s, d, sn = int(parcalar[0]), int(parcalar[1]), float(parcalar[2])
    return s * 3600.0 + d * 60.0 + sn


def _xy(loc):
    """StatsBomb konum listesinden (x, y) döndürür, isteğe bağlı z'yi yoksayar."""
    if loc is None:
        return (None, None)
    if isinstance(loc, list) and len(loc) >= 2:
        return (float(loc[0]), float(loc[1]))
    return (None, None)


def _satir_aksiyona(satir: dict, mac_id: int) -> dict | None:
    """Tek bir olay kaydını SPADL-lite aksiyon sözlüğüne dönüştürür.

    Olay AKSIYON_TURLERI'nden biri değilse veya koordinatlar eksikse None döner.
    """
    etur = satir.get("type")
    if etur not in AKSIYON_TURLERI:
        return None

    bx, by = _xy(satir.get("location"))
    if bx is None:
        return None

    if etur == "Pass":
        ex, ey = _xy(satir.get("pass_end_location"))
        # `pass_outcome` yalnızca pas başarısız olduğunda null değildir
        # (Tamamlanmadı, Dışarı, Ofsayt, vb.). Null == başarı.
        basarili = satir.get("pass_outcome") in (None, "")
    elif etur == "Shot":
        ex, ey = _xy(satir.get("shot_end_location"))
        # Yalnızca golü "başarı" say; diğerleri (kurtarış, blok, isabet dışı) başarısız.
        basarili = satir.get("shot_outcome") == "Goal"
    else:  # Taşıma
        ex, ey = _xy(satir.get("carry_end_location"))
        # Taşımaların sonucu yoktur — başarı olarak değerlendiriyoruz.
        basarili = True

    if ex is None:
        return None

    return {
        "match_id": int(mac_id),
        "action_id": satir.get("id"),
        "period": int(satir.get("period", 0)),
        "time_seconds": _zaman_damgasi_saniyeye(satir.get("timestamp", "00:00:00.000")),
        "team_name": satir.get("team"),
        "player_name": satir.get("player"),
        "type": etur.lower(),
        "start_x": bx,
        "start_y": by,
        "end_x": ex,
        "end_y": ey,
        "success": bool(basarili),
    }


def events_file_to_actions(events_path: Path, match_id: int) -> pd.DataFrame:
    """Önbelleğe alınmış bir olaylar JSON dosyasını okur ve aksiyon DataFrame'i döndürür."""
    kayitlar = json.loads(Path(events_path).read_text(encoding="utf-8"))
    satirlar = [a for r in kayitlar if (a := _satir_aksiyona(r, match_id)) is not None]
    return pd.DataFrame.from_records(satirlar)


def competition_actions(comp_dir: Path) -> pd.DataFrame:
    """`comp_dir` içindeki tüm `<mac_id>_events.json` dosyalarını okur ve birleştirir.

    Maç ID'leri dosya adı önekinden elde edilir.
    """
    cerceveler = []
    for events_path in sorted(Path(comp_dir).glob("*_events.json")):
        mac_id_str = events_path.name.split("_", 1)[0]
        try:
            mac_id = int(mac_id_str)
        except ValueError:
            continue
        cerceveler.append(events_file_to_actions(events_path, mac_id))
    if not cerceveler:
        return pd.DataFrame(
            columns=[
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
            ]
        )
    return pd.concat(cerceveler, ignore_index=True)

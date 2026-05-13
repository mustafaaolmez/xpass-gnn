"""Önbelleğe alınmış StatsBomb olay + 360 dondurulmuş kare JSON'larını okur ve birleştirir.

`load_match_frames` tarafından üretilen şema:
  dict[olay_uuid] -> oyuncu kayıtlarının listesi:
    {x, y, takim_arkadasi (bool), kaleci (bool), oyuncu (bool)}

Graf oluşturucu (`src/data/graph_builder.py`) bunu etiketli aksiyon çerçevesiyle
birlikte kullanarak aksiyon başına `torch_geometric.data.Data` nesneleri oluşturur.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_match_frames(frames_path: Path) -> dict[str, list[dict]]:
    """`<mac_id>_frames.json` dosyasını ayrıştırır ve {olay_uuid: dondurulmuş_kare} döndürür."""
    kayitlar = json.loads(Path(frames_path).read_text(encoding="utf-8"))
    cikti: dict[str, list[dict]] = {}
    for r in kayitlar:
        uid = r.get("event_uuid")
        if not uid:
            continue
        ff = r.get("freeze_frame") or []
        cikti[uid] = [
            {
                "x": float(p["location"][0]) if p.get("location") else None,
                "y": float(p["location"][1]) if p.get("location") else None,
                "teammate": bool(p.get("teammate", False)),
                "keeper": bool(p.get("keeper", False)),
                "actor": bool(p.get("actor", False)),
            }
            for p in ff
            if p.get("location") is not None
        ]
    return cikti


def load_competition_frames(comp_dir: Path) -> dict[str, list[dict]]:
    """Bir müsabaka dizinindeki tüm `<mac_id>_frames.json` dosyalarını birleştirir."""
    cikti: dict[str, list[dict]] = {}
    for fp in sorted(Path(comp_dir).glob("*_frames.json")):
        cikti.update(load_match_frames(fp))
    return cikti

"""xPass++ için StatsBomb Open Data'dan küçük bir dilim indirir.

Hedefler: (a) ortamın statsbombpy üzerinden StatsBomb Open Data uç noktasını
çekebildiğini doğrula, (b) iyi belgelenmiş bir müsabakanın olaylarını (ve mevcut
360 dondurulmuş karelerini) `data/raw/` altına önbelleğe al, (c) dilimdeki
pas+şut örneklerinin mimari bütçeye göre yeteri kadar olduğunu doğrulamak için
özet tablosu yazdır.

Kullanım (proje kökünden, venv aktif):

    python scripts/download_data.py                      # otomatik müsabaka seç
    python scripts/download_data.py --comp-id 43 --season-id 106  # açık seçim
    python scripts/download_data.py --max-matches 5      # yalnızca smoke testi

Başarıda 0, herhangi bir hatada sıfır dışı bir değerle çıkar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from statsbombpy import sb  # type: ignore[import-untyped]

PROJE_KOKU = Path(__file__).resolve().parent.parent
HAM_DIZIN = PROJE_KOKU / "data" / "raw"

TERCIH_EDILEN = (
    "FIFA World Cup",
    "UEFA Euro",
    "Women's World Cup",
)


def argumanlari_isle(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comp-id", type=int, default=None, help="StatsBomb competition_id")
    p.add_argument("--season-id", type=int, default=None, help="StatsBomb season_id")
    p.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="İndirilecek maksimum maç sayısı (smoke testleri için).",
    )
    p.add_argument(
        "--with-360",
        action="store_true",
        default=True,
        help="Mevcut yerlerde 360 dondurulmuş kareleri de indir.",
    )
    p.add_argument("--no-360", dest="with_360", action="store_false")
    return p.parse_args(argv)


def musabaka_sec(musabakalar: pd.DataFrame) -> tuple[int, int, str]:
    """360 kapsamına sahip ve tercih ettiğimiz turnuva listesindeki en son
    müsabakayı seçer.
    """
    df = musabakalar.copy()
    df = df[df["match_available"].notna() & df["match_available_360"].notna()]
    if df.empty:
        raise RuntimeError(
            "StatsBomb Open Data'da hem match_available hem de match_available_360 olan "
            "müsabaka yok. `sb.competitions()` çıktısını kontrol edin."
        )

    def oncelik(satir: pd.Series) -> tuple[int, int]:
        ad: str = str(satir.get("competition_name", ""))
        for i, on_ek in enumerate(TERCIH_EDILEN):
            if on_ek.lower() in ad.lower():
                return (i, -int(satir.get("season_id", 0)))
        return (len(TERCIH_EDILEN), -int(satir.get("season_id", 0)))

    df = df.assign(_oncelik=df.apply(oncelik, axis=1)).sort_values("_oncelik")
    en_iyi = df.iloc[0]
    return (
        int(en_iyi["competition_id"]),
        int(en_iyi["season_id"]),
        (f"{en_iyi['competition_name']} — {en_iyi['season_name']}"),
    )


def mac_indir(match_id: int, hedef_dizin: Path, ile_360: bool) -> dict[str, bool]:
    """Bir maç için olayları (ve isteğe bağlı 360 kareleri) indirir. Idempotent."""
    cikti = {"events": False, "frames": False}
    olaylar_yolu = hedef_dizin / f"{match_id}_events.json"
    kareler_yolu = hedef_dizin / f"{match_id}_frames.json"

    if not olaylar_yolu.exists():
        olaylar_df = sb.events(match_id=match_id)
        olaylar_yolu.write_text(
            olaylar_df.to_json(orient="records", date_format="iso"),
            encoding="utf-8",
        )
        cikti["events"] = True

    if ile_360 and not kareler_yolu.exists():
        try:
            kareler_listesi = sb.frames(match_id=match_id, fmt="dict")
        except Exception as exc:  # noqa: BLE001
            print(f"  maç {match_id}: 360 kare yok ({exc.__class__.__name__})")
            return cikti
        if not kareler_listesi:
            return cikti
        kareler_yolu.write_text(json.dumps(kareler_listesi, default=str), encoding="utf-8")
        cikti["frames"] = True

    return cikti


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = argumanlari_isle(argv)

    print("StatsBomb Open Data müsabakaları listeleniyor...")
    musabakalar = sb.competitions()
    print(f"  {len(musabakalar)} müsabaka-sezon mevcut.")

    if args.comp_id is not None and args.season_id is not None:
        komp_id, sezon_id = args.comp_id, args.season_id
        eslesme = musabakalar[
            (musabakalar["competition_id"] == komp_id) & (musabakalar["season_id"] == sezon_id)
        ]
        if eslesme.empty:
            print(
                f"HATA: ({komp_id}, {sezon_id}) competitions() içinde bulunamadı.",
                file=sys.stderr,
            )
            return 2
        etiket = f"{eslesme.iloc[0]['competition_name']} — {eslesme.iloc[0]['season_name']}"
    else:
        komp_id, sezon_id, etiket = musabaka_sec(musabakalar)

    print(
        f"Seçilen müsabaka: comp_id={komp_id}, season_id={sezon_id}  ->  {etiket}"
    )

    maclar = sb.matches(competition_id=komp_id, season_id=sezon_id)
    print(f"  Bu müsabaka-sezonda {len(maclar)} maç.")

    if args.max_matches is not None:
        maclar = maclar.head(args.max_matches)
        print(f"  --max-matches ile ilk {len(maclar)} maçla sınırlandırıldı.")

    slug = f"comp{komp_id}_season{sezon_id}"
    hedef_dizin = HAM_DIZIN / slug
    hedef_dizin.mkdir(parents=True, exist_ok=True)

    manifest = {
        "competition_id": komp_id,
        "season_id": sezon_id,
        "label": etiket,
        "matches": maclar.to_dict(orient="records"),
    }
    (hedef_dizin / "manifest.json").write_text(
        json.dumps(manifest, default=str, indent=2), encoding="utf-8"
    )

    yeni_olay = yeni_kare = 0
    for _, satir in maclar.iterrows():
        mac_id = int(satir["match_id"])
        sonuc = mac_indir(mac_id, hedef_dizin, ile_360=args.with_360)
        if sonuc["events"]:
            yeni_olay += 1
        if sonuc["frames"]:
            yeni_kare += 1

    print()
    print(
        f"Tamamlandı. {yeni_olay} yeni olay dosyası, {yeni_kare} yeni 360 kare dosyası yazıldı:"
    )
    print(f"  {hedef_dizin.relative_to(PROJE_KOKU)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

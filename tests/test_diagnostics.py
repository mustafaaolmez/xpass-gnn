"""Smoke kontrolü `src.diagnostics`: veri şemasının kararlı olduğunu doğrular."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJE_KOKU = Path(__file__).resolve().parent.parent
KONTROL_NOKTASI = PROJE_KOKU / "models" / "xpass_gnn" / "sprint2" / "best.pt"
DS_ONBELLEGI = PROJE_KOKU / "data" / "processed" / "xpass_dataset" / "tam"
ETIKETLER = PROJE_KOKU / "data" / "processed" / "wc2022_xt_labels.parquet"
KARELER = PROJE_KOKU / "data" / "raw" / "comp43_season106"


@pytest.mark.skipif(
    not (KONTROL_NOKTASI.exists() and ETIKETLER.exists() and KARELER.exists() and DS_ONBELLEGI.exists()),
    reason="Sprint-2 çıktıları gereklidir (veri seti önbelleği + eğitilmiş kontrol noktası).",
)
def test_tanilama_veri_semasi(tmp_path: Path) -> None:
    """run() çıktısının beklenen anahtarlara sahip olduğunu doğrular."""
    from src.diagnostics import run

    veri = run(
        labels_parquet=ETIKETLER,
        frames_dir=KARELER,
        ds_cache=DS_ONBELLEGI,
        ckpt_path=KONTROL_NOKTASI,
        out_path=tmp_path / "tanilama.json",
    )

    for k in (
        "gnn_bolum_bazi",
        "temel_bolum_bazi",
        "test_aksiyon_tur_bazi",
        "artik",
        "hipotez_kontrolu",
    ):
        assert k in veri, f"eksik üst düzey anahtar: {k}"

    for bolum in ("train", "val", "test"):
        assert bolum in veri["gnn_bolum_bazi"]
        for m in ("mae", "mse", "spearman"):
            assert m in veri["gnn_bolum_bazi"][bolum]

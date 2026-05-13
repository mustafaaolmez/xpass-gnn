"""Smoke kontrolü: `notebooks/demo_dqs.py` baştan sona çalışır ve PNG yazar."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJE_KOKU = Path(__file__).resolve().parent.parent
SCRIPT = PROJE_KOKU / "notebooks" / "demo_dqs.py"
CIKTI_PNG = PROJE_KOKU / "docs" / "figures" / "demo_dqs_pitch.png"
KONTROL_NOKTASI = PROJE_KOKU / "models" / "xpass_gnn" / "sprint4" / "best.pt"
DS_ONBELLEGI = PROJE_KOKU / "data" / "processed" / "xpass_dataset" / "tam"


@pytest.mark.skipif(
    not (KONTROL_NOKTASI.exists() and DS_ONBELLEGI.exists()),
    reason="Sprint-4 kontrol noktası + önbelleğe alınmış PyG veri seti gereklidir.",
)
def test_demo_calisir() -> None:
    sonuc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=PROJE_KOKU,
        check=False,
    )
    assert sonuc.returncode == 0, (
        f"demo_dqs.py {sonuc.returncode} ile çıktı\n"
        f"stdout: {sonuc.stdout}\nstderr: {sonuc.stderr}"
    )
    assert CIKTI_PNG.exists()
    assert CIKTI_PNG.stat().st_size > 5_000  # PNG önemsiz olmamalıdır

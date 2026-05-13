"""Smoke kontrolü: `scripts/make_figures.py` çalışır ve beklenen PNG'leri yazar."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJE_KOKU = Path(__file__).resolve().parent.parent
SCRIPT = PROJE_KOKU / "scripts" / "make_figures.py"
GRAFIK_DIZINI = PROJE_KOKU / "docs" / "figures"
S2_GECMIS = PROJE_KOKU / "models" / "xpass_gnn" / "sprint2" / "history.json"
S4_GECMIS = PROJE_KOKU / "models" / "xpass_gnn" / "sprint4" / "history.json"


@pytest.mark.skipif(
    not (S2_GECMIS.exists() and S4_GECMIS.exists()),
    reason="Sprint-2 ve Sprint-4 history JSON'ları gereklidir.",
)
def test_grafik_olusturma_calisir() -> None:
    sonuc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=PROJE_KOKU,
        check=False,
    )
    assert sonuc.returncode == 0, (
        f"make_figures.py {sonuc.returncode} ile çıktı\n"
        f"stdout: {sonuc.stdout}\nstderr: {sonuc.stderr}"
    )
    assert (GRAFIK_DIZINI / "training_curves.png").exists()
    assert (GRAFIK_DIZINI / "mae_comparison.png").exists()
    assert (GRAFIK_DIZINI / "training_curves.png").stat().st_size > 2_000
    assert (GRAFIK_DIZINI / "mae_comparison.png").stat().st_size > 2_000

"""Sprint-0 smoke testi.

Her üst düzey proje modülünü içe aktararak paket düzeninin import edilebilir
olduğunu onaylar, ardından `scripts/verify_env.py` dosyasını subprocess olarak
çalıştırır ve 0 ile çıktığını doğrular.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJE_KOKU = Path(__file__).resolve().parent.parent
ORTAM_DOGRULA = PROJE_KOKU / "scripts" / "verify_env.py"


def test_src_paketi_import_edilebilir() -> None:
    """Tüm `src/**/__init__.py` paketleri repo kökünden import edilebilmelidir."""
    if str(PROJE_KOKU) not in sys.path:
        sys.path.insert(0, str(PROJE_KOKU))
    import src  # noqa: F401
    import src.data  # noqa: F401
    import src.models  # noqa: F401
    import src.utils  # noqa: F401


def test_ortam_dogrulama_calisir() -> None:
    """`scripts/verify_env.py` aktif yorumlayıcıda 0 ile çıkmalıdır."""
    if not ORTAM_DOGRULA.exists():
        pytest.fail(f"verify_env.py bulunamadı: {ORTAM_DOGRULA}")

    sonuc = subprocess.run(
        [sys.executable, str(ORTAM_DOGRULA)],
        capture_output=True,
        text=True,
        cwd=PROJE_KOKU,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )

    assert sonuc.returncode == 0, (
        f"verify_env.py {sonuc.returncode} ile çıktı\n"
        f"--- stdout ---\n{sonuc.stdout}\n"
        f"--- stderr ---\n{sonuc.stderr}\n"
    )

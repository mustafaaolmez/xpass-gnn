"""xPass++ geliştirme ortamını doğrular.

Her sabitlenmiş bağımlılığı içe aktarır, kurulu sürümünü yazdırır ve
PyTorch için CUDA destekli GPU'nun mevcut olduğunu doğrular.

Proje kökünden çalıştır:

    python scripts/verify_env.py

Başarıda 0, başarısızlıkta sıfır dışı bir değerle çıkar.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import importlib.metadata as _md
import sys
from pathlib import Path

PAKETLER: dict[str, str] = {
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "statsbombpy": "statsbombpy",
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "mplsoccer": "mplsoccer",
    "pytest": "pytest",
    "ruff": "ruff",
    "black": "black",
    "ipykernel": "ipykernel",
}

PROJE_KOKU = Path(__file__).resolve().parent.parent


def _surum(import_adi: str, dist_adi: str) -> str:
    """Paketin kurulu sürümünü döndürür, bulunamazsa '<eksik>'."""
    try:
        return _md.version(dist_adi)
    except _md.PackageNotFoundError:
        try:
            mod = importlib.import_module(import_adi)
            return getattr(mod, "__version__", "<__version__ yok>")
        except ImportError:
            return "<eksik>"


def paketleri_kontrol_et() -> tuple[dict[str, str], list[str]]:
    """(import_adina_gore_surumler, eksik_dist_adlari) döndürür."""
    surumler: dict[str, str] = {}
    eksikler: list[str] = []
    for import_adi, dist_adi in PAKETLER.items():
        s = _surum(import_adi, dist_adi)
        surumler[import_adi] = s
        if s == "<eksik>":
            eksikler.append(dist_adi)
    return surumler, eksikler


def cuda_kontrol() -> dict[str, object]:
    """PyTorch'ta CUDA + cihaz bilgisini sorgular. CUDA yoksa AssertionError fırlatır."""
    import torch

    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() False döndürdü. "
        "CUDA tekerini kurdunuz mu? Talimatlar için requirements.txt başlığına bakın."
    )

    cihaz_sayisi = torch.cuda.device_count()
    bilgi: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_count": cihaz_sayisi,
        "devices": [],
    }
    for i in range(cihaz_sayisi):
        ozellikler = torch.cuda.get_device_properties(i)
        bilgi["devices"].append(
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(ozellikler.total_memory / 1e9, 2),
                "compute_capability": "%d.%d" % torch.cuda.get_device_capability(i),
                "multi_processor_count": ozellikler.multi_processor_count,
            }
        )
    return bilgi


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print()

    print("Kurulu paket sürümleri:")
    surumler, eksikler = paketleri_kontrol_et()
    for ad, sur in surumler.items():
        print(f"  {ad:18s} {sur}")
    print()

    if eksikler:
        print(f"HATA: eksik paketler: {eksikler}", file=sys.stderr)
        return 2

    try:
        cuda_bilgi = cuda_kontrol()
    except AssertionError as exc:
        print(f"HATA: CUDA kontrolü başarısız — {exc}", file=sys.stderr)
        return 3

    print("CUDA / GPU:")
    print(f"  torch CUDA çalışma zamanı: {cuda_bilgi['cuda_runtime']}")
    print(f"  {cuda_bilgi['device_count']} CUDA cihazı")
    for d in cuda_bilgi["devices"]:  # type: ignore[union-attr]
        print(
            f"  - [{d['index']}] {d['name']} — "
            f"{d['total_memory_gb']} GB, hesaplama {d['compute_capability']}, "
            f"{d['multi_processor_count']} SM"
        )
    print()
    print("Ortam doğrulaması başarıyla tamamlandı.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

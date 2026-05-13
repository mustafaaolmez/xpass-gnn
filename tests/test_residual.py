"""Artık kafası hattı için smoke kontrolü.

`precompute_baseline_preds` çıktısının veri seti uzunluğuyla eşleştiğini ve
`_rewrite_labels_to_residuals`'in etiket std sapmasını gerçekten küçülttüğünü doğrular
(artıklar ham etiketlerden daha düşük varyanslı olmalıdır — temelin amacı budur).
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJE_KOKU = Path(__file__).resolve().parent.parent
ETIKETLER = PROJE_KOKU / "data" / "processed" / "wc2022_xt_labels.parquet"
KARELER = PROJE_KOKU / "data" / "raw" / "comp43_season106"
DS_ONBELLEGI = PROJE_KOKU / "data" / "processed" / "xpass_dataset" / "tam"


@pytest.mark.skipif(
    not (ETIKETLER.exists() and KARELER.exists() and DS_ONBELLEGI.exists()),
    reason="WC2022 etiket parquet + 360 kare + veri seti önbelleği gereklidir.",
)
def test_temel_hizalama_ve_artik_varyans_dusus() -> None:
    from src.data.dataset import XPassDataset
    from src.train_residual import (
        _rewrite_labels_to_residuals,
        precompute_baseline_preds,
    )

    ds = XPassDataset(labels_parquet=ETIKETLER, frames_dir=KARELER, cache_dir=DS_ONBELLEGI)
    n = len(ds)
    tahminler = precompute_baseline_preds(KARELER, n_expected=n)
    assert tahminler.shape == (n,)
    assert tahminler.dtype.kind == "f"

    y_onceki = ds._data.y.detach().clone()
    std_onceki = float(y_onceki.std().item())

    _rewrite_labels_to_residuals(ds, tahminler)
    y_sonraki = ds._data.y.detach()
    std_sonraki = float(y_sonraki.std().item())

    # Artıklar ham etiketlerden daha düşük std'ye sahip olmalıdır.
    assert (
        std_sonraki < std_onceki
    ), f"artık std ({std_sonraki}) ham etiket std'nden ({std_onceki}) küçük olmalıdır"

    # Sonraki testlerin bellekteki temsili bozulmasın diye ham etiketleri geri yükle.
    ds._data.y = y_onceki

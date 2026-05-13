"""xT-delta etiket üreticisi — Karun Singh (2018) Beklenen Tehdit.

Referans: Singh, K. "Introducing Expected Threat (xT)." https://karun.in/blog/expected-threat.html

Bu modül:
  1) Aksiyonları 16 × 12 ızgara üzerine yerleştirir (StatsBomb sahası 120 × 80 →
     bölge boyutu 7.5 × 6.667; Singh'in geometrisiyle eşleşir).
  2) s, m, g, T değerlerini aksiyon corpus'undan tahmin eder.
  3) xT'yi yakınsamaya kadar yineler.
  4) Her aksiyon için xT-delta döndürür:
       başarılı pas / taşıma: xT(bitiş) − xT(başlangıç)
       başarısız pas / taşıma: −xT(başlangıç)   (top kaybı)
       şut: g(başlangıç) − xT(başlangıç)         (sahip olunan pozisyon kullanıldı)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Saha boyutları ve ızgara geometrisi. StatsBomb açık veri 120 × 80 kullanır.
SAHA_UZUNLUK = 120.0
SAHA_GENISLIK = 80.0
VARSAYILAN_NX = 16
VARSAYILAN_NY = 12

# Geriye dönük uyumluluk için dışa aktarılan takma adlar
PITCH_LEN = SAHA_UZUNLUK
PITCH_WID = SAHA_GENISLIK
DEFAULT_NX = VARSAYILAN_NX
DEFAULT_NY = VARSAYILAN_NY


def _zone_of(
    x: np.ndarray, y: np.ndarray, n_x: int, n_y: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vektörleştirilmiş (x, y) → (zx, zy). [0, n_x-1] × [0, n_y-1] aralığına kırpar."""
    zx = np.clip(np.floor(x / SAHA_UZUNLUK * n_x).astype(int), 0, n_x - 1)
    zy = np.clip(np.floor(y / SAHA_GENISLIK * n_y).astype(int), 0, n_y - 1)
    return zx, zy


class XTDeltaLabels:
    """Singh xT etiket üreticisi. Önce :meth:`fit`, sonra :meth:`compute` çağrılır.

    Fit sonrası nitelikler:
        xt: (n_x, n_y) dizi, yakınsanmış xT ızgarası.
        s, m, g: (n_x, n_y) bölge başına marjinal aksiyon olasılıkları.
        T: (n_x, n_y, n_x, n_y) hareket geçiş tensörü.
        iterations: int, yakınsamaya kadar iterasyon sayısı.
        residual: float, sabit nokta güncellemesinin son L_inf artığı.
    """

    name = "xt_delta"

    def __init__(
        self,
        n_x: int = VARSAYILAN_NX,
        n_y: int = VARSAYILAN_NY,
        max_iter: int = 100,
        tol: float = 1e-5,
    ) -> None:
        self.n_x = n_x
        self.n_y = n_y
        self.max_iter = max_iter
        self.tol = tol

        self.xt: np.ndarray | None = None
        self.s: np.ndarray | None = None
        self.m: np.ndarray | None = None
        self.g: np.ndarray | None = None
        self.T: np.ndarray | None = None
        self.iterations: int = 0
        self.residual: float = float("nan")

    # ------------------------------------------------------------------
    # Fit

    def fit(self, actions: pd.DataFrame) -> "XTDeltaLabels":
        """Aksiyon çerçevesinden s, m, g, T tahmin eder ve xT'yi yakınsamaya kadar yineler.

        Beklenen sütunlar: type (pass/shot/carry), start_x, start_y, end_x, end_y, success.
        """
        n_x, n_y = self.n_x, self.n_y

        hareketler = actions[actions["type"].isin(("pass", "carry"))]
        sutlar = actions[actions["type"] == "shot"]

        bx_h, by_h = _zone_of(
            hareketler["start_x"].to_numpy(), hareketler["start_y"].to_numpy(), n_x, n_y
        )
        ex_h, ey_h = _zone_of(
            hareketler["end_x"].to_numpy(), hareketler["end_y"].to_numpy(), n_x, n_y
        )
        bx_s, by_s = _zone_of(
            sutlar["start_x"].to_numpy(), sutlar["start_y"].to_numpy(), n_x, n_y
        )

        # Bölge başına sayım.
        hareket_sayisi = np.zeros((n_x, n_y))
        sut_sayisi = np.zeros((n_x, n_y))
        gol_sayisi = np.zeros((n_x, n_y))
        hareket_basarili = hareketler["success"].to_numpy(dtype=bool)
        for i in range(len(hareketler)):
            hareket_sayisi[bx_h[i], by_h[i]] += 1
        for i in range(len(sutlar)):
            sut_sayisi[bx_s[i], by_s[i]] += 1
            if sutlar.iloc[i]["success"]:
                gol_sayisi[bx_s[i], by_s[i]] += 1

        # Marjinaller s, m, g.
        aksiyon_sayisi = hareket_sayisi + sut_sayisi
        # NaN üretmemek için Laplace düzleştirme.
        s = np.where(aksiyon_sayisi > 0, sut_sayisi / np.maximum(aksiyon_sayisi, 1), 0.0)
        m = 1.0 - s
        g = np.where(sut_sayisi > 0, gol_sayisi / np.maximum(sut_sayisi, 1), 0.0)

        # Geçiş tensörü T[bx, by, ex, ey] başarılı hareketlerden.
        T = np.zeros((n_x, n_y, n_x, n_y))
        tb_bx, tb_by = bx_h[hareket_basarili], by_h[hareket_basarili]
        tb_ex, tb_ey = ex_h[hareket_basarili], ey_h[hareket_basarili]
        for i in range(tb_bx.shape[0]):
            T[tb_bx[i], tb_by[i], tb_ex[i], tb_ey[i]] += 1.0
        satir_toplam = T.sum(axis=(2, 3), keepdims=True)
        T_normalize = np.zeros_like(T)
        np.divide(T, satir_toplam, where=satir_toplam > 0, out=T_normalize)
        T = T_normalize

        # Sabit nokta yinelemesi.
        xt = np.zeros((n_x, n_y))
        onceki = xt.copy()
        for it in range(1, self.max_iter + 1):
            hareket_degeri = np.einsum("xyzw,zw->xy", T, onceki)
            xt = s * g + m * hareket_degeri
            artik = float(np.max(np.abs(xt - onceki)))
            onceki = xt
            if artik < self.tol:
                self.iterations = it
                self.residual = artik
                break
        else:
            self.iterations = self.max_iter
            self.residual = artik

        self.xt = xt
        self.s = s
        self.m = m
        self.g = g
        self.T = T
        return self

    # ------------------------------------------------------------------
    # Hesapla

    def _lookup(self, x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
        zx, zy = _zone_of(x, y, self.n_x, self.n_y)
        return grid[zx, zy]

    def compute(self, actions: pd.DataFrame) -> pd.Series:
        """Aksiyon başına xT-delta. Önceden :meth:`fit` çağrılmalıdır."""
        if self.xt is None:
            raise RuntimeError("XTDeltaLabels: .compute() çağrısından önce .fit(actions) çağrın.")

        xt = self.xt
        g = self.g

        baslangic_xt = self._lookup(
            actions["start_x"].to_numpy(), actions["start_y"].to_numpy(), xt
        )
        bitis_xt = self._lookup(
            actions["end_x"].to_numpy(), actions["end_y"].to_numpy(), xt
        )
        sut_g = self._lookup(
            actions["start_x"].to_numpy(), actions["start_y"].to_numpy(), g
        )

        turler = actions["type"].to_numpy()
        basarili = actions["success"].to_numpy(dtype=bool)

        cikti = np.zeros(len(actions))
        hareket_mi = (turler == "pass") | (turler == "carry")
        cikti = np.where(hareket_mi & basarili, bitis_xt - baslangic_xt, cikti)
        cikti = np.where(hareket_mi & ~basarili, -baslangic_xt, cikti)
        sut_mu = turler == "shot"
        cikti = np.where(sut_mu, sut_g - baslangic_xt, cikti)

        return pd.Series(cikti, index=actions.index, name=self.name)

"""`docs/figures/` dizinine iki statik grafik üretir.

Her iki grafik de önbelleğe alınmış Sprint-2 ve Sprint-4 geçmiş/metrik
JSON dosyalarından tekrar üretilebilir; model yeniden yüklenmesine gerek yoktur.

Çıktılar:
  - `training_curves.png` — epoch boyunca egitim/dogrulama kaybı, Sprint 2 + Sprint 4
  - `mae_comparison.png` — {xT-temeli, Sprint-2 GAT, Sprint-4 DQS} için test MAE çubuk grafiği

Proje kökünden çalıştır:

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # başsız arka uç — CI / scriptler için görüntü gerekmez
import matplotlib.pyplot as plt  # noqa: E402

PROJE_KOKU = Path(__file__).resolve().parent.parent
S2_GECMIS = PROJE_KOKU / "models" / "xpass_gnn" / "sprint2" / "history.json"
S4_GECMIS = PROJE_KOKU / "models" / "xpass_gnn" / "sprint4" / "history.json"
S2_METRIK = PROJE_KOKU / "data" / "processed" / "sprint2_metrics.json"
S4_METRIK = PROJE_KOKU / "data" / "processed" / "sprint4_metrics.json"
GRAFIK_DIZINI = PROJE_KOKU / "docs" / "figures"


def _egitim_egrisi(s2: dict, s4: dict, cikti_yolu: Path) -> None:
    fig, eksenler = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    # Sprint 2 paneli.
    ax = eksenler[0]
    epochlar_s2 = [r["epoch"] for r in s2["history"]]
    ax.plot(epochlar_s2, [r["train_loss"] for r in s2["history"]], label="egitim")
    ax.plot(epochlar_s2, [r["val_loss"] for r in s2["history"]], label="dogrulama")
    ax.axvline(
        s2["best_epoch"],
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"en iyi epoch ({s2['best_epoch']})",
    )
    ax.set_title("Sprint 2 — ham etiket GAT")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Smooth-L1 kaybı")
    ax.legend()
    ax.grid(alpha=0.3)

    # Sprint 4 paneli.
    ax = eksenler[1]
    epochlar_s4 = [r["epoch"] for r in s4["history"]]
    ax.plot(epochlar_s4, [r["train_loss"] for r in s4["history"]], label="egitim")
    ax.plot(epochlar_s4, [r["val_loss"] for r in s4["history"]], label="dogrulama")
    ax.axvline(
        s4["best_epoch"],
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"en iyi epoch ({s4['best_epoch']})",
    )
    ax.set_title("Sprint 4 — artık kafası GAT")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Smooth-L1 kaybı (artık ölçek)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Egitim eğrileri: ham etiket - artık kafası GAT karşılaştırması")
    fig.savefig(cikti_yolu, dpi=140)
    plt.close(fig)


def _mae_karsilastirmasi(s2_metrik: dict, s4_metrik: dict, cikti_yolu: Path) -> None:
    etiketler = [
        "xT-temeli",
        "Sprint-2 GAT\n(ham etiket)",
        "Sprint-4 DQS\n(temel + artık)",
    ]
    maeler = [
        s2_metrik["baseline_xt"]["mae"],
        s2_metrik["gnn"]["mae"],
        s4_metrik["final_dqs"]["mae"],
    ]
    renkler = ["#888888", "#cc4444", "#3a78c4"]

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    cubuklar = ax.bar(etiketler, maeler, color=renkler)
    for cubuk, deger in zip(cubuklar, maeler, strict=False):
        ax.text(
            cubuk.get_x() + cubuk.get_width() / 2,
            deger + 0.0005,
            f"{deger:.5f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Test bölümü MAE (düşük = iyi)")
    ax.set_title("WC 2022 test bölümü — xT-delta hedefi")
    ax.axhline(maeler[0], color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_ylim(0, max(maeler) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(cikti_yolu, dpi=140)
    plt.close(fig)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    GRAFIK_DIZINI.mkdir(parents=True, exist_ok=True)

    s2 = json.loads(S2_GECMIS.read_text(encoding="utf-8"))
    s4 = json.loads(S4_GECMIS.read_text(encoding="utf-8"))
    s2_metrik = json.loads(S2_METRIK.read_text(encoding="utf-8"))
    s4_metrik = json.loads(S4_METRIK.read_text(encoding="utf-8"))

    egitim_yolu = GRAFIK_DIZINI / "training_curves.png"
    mae_yolu = GRAFIK_DIZINI / "mae_comparison.png"
    _egitim_egrisi(s2, s4, egitim_yolu)
    _mae_karsilastirmasi(s2_metrik, s4_metrik, mae_yolu)

    print(f"Yazıldı: {egitim_yolu.relative_to(PROJE_KOKU)}")
    print(f"Yazıldı: {mae_yolu.relative_to(PROJE_KOKU)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

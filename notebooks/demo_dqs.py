"""Demo: tutulan test bölümü maçını skorla ve DQS saha grafiğini oluştur.

Proje kökünden çalıştır:

    python notebooks/demo_dqs.py [mac_id]

`mac_id` belirtilmezse, script deterministik Sprint-2/Sprint-4 test bölümündeki
ilk maçı seçer. Çıktı:

    docs/figures/demo_dqs_pitch.png

Grafik, seçilen maçtaki her pası başlangıç konumundan bitiş konumuna ok olarak
gösterir; oklar tahmin edilen DQS değerine göre renklenir (mavi = yüksek değer,
kırmızı = düşük değer).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mplsoccer import Pitch  # type: ignore[import-untyped]  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from src.data.dataset import XPassDataset, match_grouped_split  # noqa: E402
from src.models.xpass_gnn import XPassGNN  # noqa: E402

PROJE_KOKU = Path(__file__).resolve().parent.parent
ETIKETLER = PROJE_KOKU / "data" / "processed" / "wc2022_xt_labels.parquet"
KARELER = PROJE_KOKU / "data" / "raw" / "comp43_season106"
DS_ONBELLEGI = PROJE_KOKU / "data" / "processed" / "xpass_dataset" / "tam"
KONTROL_NOKTASI = PROJE_KOKU / "models" / "xpass_gnn" / "sprint4" / "best.pt"
TEMEL_NPY = PROJE_KOKU / "models" / "xpass_gnn" / "sprint4" / "baseline_preds.npy"
CIKTI_PNG = PROJE_KOKU / "docs" / "figures" / "demo_dqs_pitch.png"


def main(mac_id: int | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ds = XPassDataset(labels_parquet=ETIKETLER, frames_dir=KARELER, cache_dir=DS_ONBELLEGI)
    bolumler = match_grouped_split(ds, seed=42)
    test_indisleri = bolumler["test"]

    # Veri seti indeksi → mac_id eşlemesi.
    graf_mac = np.asarray([int(ds[i].match_id.item()) for i in test_indisleri])
    if mac_id is None:
        mac_id = int(graf_mac[0])
    secilen_maske = graf_mac == mac_id
    if not secilen_maske.any():
        print(f"HATA: mac_id {mac_id} test bölümünde yok.", file=sys.stderr)
        return 2
    secilen_indisler = [test_indisleri[i] for i in np.where(secilen_maske)[0]]
    print(
        f"mac_id {mac_id} skorlanıyor: test bölümünde {len(secilen_indisler)} pas+şut aksiyonu."
    )

    # Artığı tahmin et, sonra temeli ekle.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XPassGNN().to(device)
    model.load_state_dict(torch.load(KONTROL_NOKTASI, weights_only=True, map_location=device))
    model.eval()

    alt_kume = ds[secilen_indisler]
    loader = DataLoader(alt_kume, batch_size=256, shuffle=False)
    ytahminler: list[np.ndarray] = []
    with torch.no_grad():
        for parti in loader:
            parti = parti.to(device)
            tahmin = model(
                parti.x, parti.edge_index, parti.edge_attr, parti.batch
            ).squeeze(-1)
            ytahminler.append(tahmin.detach().cpu().numpy())
    gat_artigi = np.concatenate(ytahminler)
    temel_hepsi = np.load(TEMEL_NPY)
    temel = temel_hepsi[np.array(secilen_indisler)]
    dqs = gat_artigi + temel
    print(
        f"  Bu maçtaki DQS aralığı: [{dqs.min():.4f}, {dqs.max():.4f}], "
        f"ortalama {dqs.mean():.4f}"
    )

    # Pas satırlarını kaynak parquet'ten al.
    import pandas as pd

    tum_aksiyonlar = pd.read_parquet(ETIKETLER, engine="fastparquet")
    aksiyonlar_ps = tum_aksiyonlar[tum_aksiyonlar["type"].isin(("pass", "shot"))].reset_index(
        drop=True
    )
    secilen_satirlar = aksiyonlar_ps.iloc[secilen_indisler].reset_index(drop=True)
    # Görselleştirme için yalnızca paslar:
    pas_mi = secilen_satirlar["type"].to_numpy() == "pass"
    pas_satirlari = secilen_satirlar.loc[pas_mi].reset_index(drop=True)
    pas_dqs = dqs[pas_mi]
    print(
        f"  sahada {len(pas_satirlari)} pas gösteriliyor (şutlar grafik dışında)."
    )

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#22312b", line_color="#c7d5cc")
    fig, ax = pitch.draw(figsize=(11, 7))
    norm = plt.Normalize(
        vmin=float(np.quantile(pas_dqs, 0.05)),
        vmax=float(np.quantile(pas_dqs, 0.95)),
    )
    cmap = plt.get_cmap("coolwarm_r")  # mavi = yüksek DQS, kırmızı = düşük
    renkler = cmap(norm(pas_dqs))

    pitch.arrows(
        pas_satirlari["start_x"].to_numpy(),
        pas_satirlari["start_y"].to_numpy(),
        pas_satirlari["end_x"].to_numpy(),
        pas_satirlari["end_y"].to_numpy(),
        width=1.4,
        headwidth=4,
        headlength=4,
        color=renkler,
        ax=ax,
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, label="DQS = temel + GAT_artigi")
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle(
        f"xPass++ Karar Kalitesi Skoru — WC 2022 maç {mac_id}\n"
        f"kontrol noktası: models/xpass_gnn/sprint4/best.pt ({len(pas_satirlari)} pas gösteriliyor)",
        fontsize=11,
    )

    CIKTI_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CIKTI_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Kaydedildi: {CIKTI_PNG.relative_to(PROJE_KOKU)}")
    return 0


if __name__ == "__main__":
    arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(main(arg))

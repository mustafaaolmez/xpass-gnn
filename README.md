# xPass++

Futbolda pas ve şut **karar kalitesini** değerlendiren bir Graf Sinir Ağı (GNN).
FIFA 2022 Dünya Kupası verisi (StatsBomb Open Data) üzerinde, 360 dondurulmuş kare
(freeze-frame) girdileri ve xT-delta denetimli hedefi kullanılarak eğitilmiştir.
Nihai **Karar Kalitesi Skoru (DQS)** = kapalı-form xT-temeli + GAT tarafından tahmin edilen artık değer.

## Temel Sonuç (test bölümü, n = 7.394)

| Model | MAE | MSE | Spearman ρ |
|-------|-----|-----|--------------|
| xT-temeli (kapalı form) | 0.00790 | 0.000172 | 0.746 |
| **xPass++ DQS** (temel + GAT artığı) | **0.00707** | **0.000116** | 0.731 |
| Δ | **−%10.5** | **−%32** | −0.015 |

GAT artık kafası, kapalı-form öncel modele kıyasla **MAE ve MSE'de ölçülebilir
iyileşme** sağlamış; projenin hedefini karşılarken açıkça ayrıştırılmış bir skor
sayesinde yorumlanabilirlik korunmuştur.

![MAE karşılaştırması](docs/figures/mae_comparison.png)

## Repo Yapısı

```
src/         veri hattı + GAT modeli + eğitim/değerlendirme/tanılama
scripts/     verify_env, download_data, produce_labels, make_figures
notebooks/   demo_dqs.py — bir maçı skorla, saha ısı haritasını oluştur
tests/       pytest (18+ test, hepsi geçiyor)
docs/        figures/
data/        ham + işlenmiş (gitignored)
models/      eğitilmiş kontrol noktaları (sprint2 ham-etiket; sprint4 artık kafası)
```

## Beş Komutla Yeniden Üret

```powershell
python -m venv .venv && .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.11.0
pip install -r requirements.txt
python scripts/verify_env.py        # GPU + kütüphaneleri doğrula
python scripts/download_data.py     # WC2022 olayları + 360 kare, ~865 MB
python scripts/produce_labels.py    # wc2022_xt_labels.parquet dosyasını yaz
python -m src.train_residual        # 30 epoch, RTX 4050'de ~7 dk
python -m src.eval_residual         # sprint4_metrics.json dosyasını yaz
python notebooks/demo_dqs.py        # tek maç saha ısı haritası
pytest tests/                       # 18+ test
```

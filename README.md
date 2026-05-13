# xPass++

Futbolda pas ve şut **karar kalitesini** değerlendiren bir Graf Sinir Ağı (GNN).
FIFA 2022 Dünya Kupası verisi (StatsBomb Open Data) üzerinde, 360° dondurulmuş kare
(freeze-frame) girdileri ve xT-delta denetimli hedefi kullanılarak eğitilmiştir.
Nihai **Karar Kalitesi Skoru (DQS)** = kapalı-form xT-temeli + GAT tarafından tahmin edilen artık değer.

> 🌐 **Web Arayüzü:** [mustafaaolmez.github.io/xpass-gnn](https://mustafaaolmez.github.io/xpass-gnn/)
> 📄 **Rapor:** [gnn_xpass_rapor.pdf](./gnn_xpass_rapor.pdf)

---

## Temel Sonuç (test bölümü, n = 7.394)

| Model | MAE | MSE | Spearman ρ |
| --- | --- | --- | --- |
| xT-temeli (kapalı form) | 0.00790 | 0.000172 | 0.746 |
| Sprint-2 GAT | 0.00750 | 0.000145 | 0.738 |
| **xPass++ DQS** (temel + GAT artığı) | **0.00707** | **0.000116** | 0.731 |
| Δ (Baseline → xPass++) | **−%10.5** | **−%32** | −0.015 |

GAT artık kafası, kapalı-form öncel modele kıyasla **MAE'de %10.5, MSE'de %32** iyileşme
sağlamış; açıkça ayrıştırılmış skor sayesinde yorumlanabilirlik korunmuştur.

![MAE karşılaştırması](https://github.com/mustafaaolmez/xpass-gnn/raw/main/docs/figures/mae_comparison.png)

---

## Model Mimarisi

| Bileşen | Değer |
| --- | --- |
| Mimari | Graph Attention Network (GAT) |
| Katman Sayısı | 3 |
| Dikkat Kafası | 4 (çok başlıklı) |
| Gizli Boyut | 128 |
| Toplam Parametre | ~211.000 |
| Aktivasyon | ELU + Dropout 0.3 |
| Optimizer | AdamW, LR = 1e-3 |
| Erken Durdurma | Sabır = 15 |

---

## Veri Seti

| Özellik | Değer |
| --- | --- |
| Kaynak | StatsBomb Open Data — FIFA WC 2022 |
| Maç Sayısı | 64 |
| Graf Sayısı | 73.946 |
| Eylem Türleri | Pas + Şut |
| Düğüm Özellikleri | 16 boyut |
| Kenar Özellikleri | 8 boyut |
| Train / Val / Test | %70 / %15 / %15 |

---

## Kısıtlamalar

- **Veri kapsamı:** Model yalnızca FIFA 2022 Dünya Kupası verisiyle eğitilmiştir; lig maçlarına genellemesi test edilmemiştir.
- **Freeze-frame eksikliği:** Bazı maçlarda 360° kare verisi eksiktir; bu eylemler veri hattından çıkarılmıştır.
- **Statik graf:** Model anlık oyuncu pozisyonlarını kullanır; oyuncuların hareket vektörleri tam olarak modellenmemiştir.
- **Spearman ρ:** MAE ve MSE iyileşirken Spearman ρ hafif düşmüştür; model sıralama tutarlılığından ziyade mutlak hata minimizasyonuna odaklanmaktadır.
- **Yalnızca pas ve şut:** Diğer eylem türleri (top sürme, müdahale vb.) kapsam dışındadır.

---

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

---

## Beş Komutla Yeniden Üret

```bash
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

---

## Proje Bilgisi

**Öğrenci:** Mustafa Necati ÖLMEZ (230212030)
**Bölüm:** Yapay Zeka Mühendisliği — OSTİM Teknik Üniversitesi
**Ders:** Derin Öğrenme Final Projesi
**Danışman:** Dr. Murat ŞİMŞEK

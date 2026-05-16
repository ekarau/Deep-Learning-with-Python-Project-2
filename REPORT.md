# REPORT — Teknik Notlar (Türkçe)

Bu dosya `PAPER.md` ile paralel ilerleyen Türkçe teknik defterdir. Grup içi anlaşmaya yardımcı olacak şekilde, hocanın sunumda görmek istediği jargonu da içerir.

---

## 1. Hocanın istediği jargonun haritası

Aşağıdaki listede hocanın notlama rubriğinden çıkan kavramlar var. Sunumda **hangi mimari kararla** bu kavramı kullanacağımızı buraya bağlıyoruz.

| Hocanın jargon listesi | Bizim mimaride nerede |
|---|---|
| Learning algorithms, capacity, over/underfitting | §4.6 Regularization study — train–val gap'ı gösterir tablo |
| Bias–variance tradeoff | Aynı regularization tablosu; data augmentation'ın varyans azaltıcı etkisi |
| Estimators, MLE | Segmentasyon kaybı (cross-entropy) = log-likelihood maximization |
| Bayesian statistics | VAE branch — KL prior, posterior approximation |
| SGD, hyperparameters, validation sets | §4.4 Optimiser comparison + 5-fold CV |
| ReLU / sigmoid / tanh / softmax | Encoder = LeakyReLU; sınıflandırma = softmax; AE recon = identity/sigmoid; LSTM gate = sigmoid |
| Architecture design principles | §3.2 — her bloğun nedeni veri yapısından çıkıyor |
| Backpropagation + computational graphs | Sunumda mimari diagramı = computational graph |
| L1/L2 norm penalties | §4.6 — L₂ weight decay 1e-5; L1 sparsity AE'de bonus eklenecek |
| Dropout + probabilistic interpretation | §4.6 — dropout p=0.2; Bayesian approximate inference yorumu |
| Data augmentation, noise robustness | Augmentation pipeline (§3.1); denoising AE'nin gürültü dayanıklılığı |
| Early stopping, batch norm | §3.4 — patience 20 + InstanceNorm |
| Ensemble methods | 3 farklı seed ile ensemble = headline sonuç |
| Pure optimization vs learning | Empirical risk minimization ↔ generalization açıklaması |
| Ill-conditioning, saddle points | §4.4 — momentum'un saddle point'leri geçişine etkisi |
| Vanishing/exploding gradients | §3.2 — LSTM gate mekanizmasının gerekçesi |
| SGD with momentum, Nesterov | §4.4 |
| AdaGrad, RMSProp, Adam | §4.4 |
| Initialization (He, Xavier, zero) | §4.5 |
| Learning rate scheduling | Cosine annealing, warm-up |

**Kural:** Sunumda bir slayt = bir kavramın ampirik kanıtı (tablo veya grafik). Hoca jargonu konuşurken görmek istiyor, ama "ezberden okuma" notu sıfırlıyor — kavramı kendi grafiğimize bağlayarak konuşmalıyız.

---

## 2. Bonus kategorisi takibi

| Kategori | Hedef | Şu an |
|---|---|---|
| +15 dataset (research paper) | ACDC — Bernard 2018 IEEE TMI | ✅ |
| +15 5+ blok | 3D-CNN, Conv-AE, ConvLSTM, Attention, VAE | ✅ plan |
| +15 ablation | 7 senaryo + optimiser + init + regularization | 🔧 |
| +15 conference-style doc | `PAPER.md` | 🔧 iskelet hazır |

---

## 3. Ablation matrisinin sebebi

Hoca "her bileşeni gerekçelendirin" diyor. Ablation = bu gerekçeyi ampirik kanıta dönüştürmek.

- A1 (`− ConvLSTM`) — temporal modelin gerçekten gerekli olduğunu kanıtlar. Beklenen düşüş 0.04-0.06 Dice.
- A2 (`− AE pretraining`) — küçük veri rejiminde self-supervised başlangıcın etkisi.
- A3 (`− Attention`) — skip refinement'in etkisi.
- A4 (`− VAE`) — probabilistic latent classification katkısı.
- A5 (`2D-CNN`) — 3D bağlamının önemi; sunumda en çarpıcı düşüş muhtemelen burada.
- A6 (`LSTM → GRU`) — gate mekanizması karşılaştırması; vanishing gradient analizine bağ.
- A7 (`− augmentation`) — overfit ve generalisation ilişkisi; bias-variance grafiği için ana kaynak.

**Önemli:** Her ablation aynı 5-fold split, aynı seed, aynı epoch sayısı. Yoksa karşılaştırma sahte.

---

## 4. Veri ve compute bütçesi

| Aşama | Tahmini süre (Colab T4) |
|---|---|
| İndirme + preprocessing | 30 dk |
| AE pretraining (50 epoch) | 2 saat |
| Full model eğitimi (100 epoch) | 3-4 saat |
| 7 ablation × ~2 saat (5 kişi paralel) | wallclock ~3 saat |
| Hyperparameter sweep (optimiser × 4) | 4 × 2 saat = 8 saat |
| **Proje toplam aktif compute** | **~16-20 saat** |

5 kişi paralel hesaplarsa 1 hafta içinde rahat biter.

---

## 5. Sunum (3 dakika) için iskelet

Hocanın 30/40/30 notlamasını hatırla.

| Saniye | İçerik | Görsel |
|---|---|---|
| 0:00–0:20 | Problem + dataset (ACDC kim, niye) | ACDC örnek görsel + sınıf dağılımı |
| 0:20–1:00 | Mimari + 5 blok gerekçesi | architecture.png |
| 1:00–1:40 | Optimiser/regularization/init kararları | §4.4-4.6 tabloları |
| 1:40–2:30 | Ablation tablosu + en önemli 2 bulgu | §4.3 tablosu vurgulu |
| 2:30–3:00 | Sonuç + sınırlamalar + ileri yön | tek slayt özet |

Sunumcu **tek kişi** (en akıcı konuşan). Diğerleri Q&A'da.

---

## 6. Açık riskler

1. **Colab oturum kopması** — checkpoint her epoch sonunda Drive'a yazılmalı.
2. **AE pretraining'in eklenmesi karmaşık** — Stage A baseline'ı bittikten sonra ekle, üzerine ablation yap. Aksi takdirde başlangıçtan boğulur.
3. **VAE diagnosis branch'ı opsiyonel görünüyor** — 5+ blok için *gerekli*. Eğer çıkarırsak 4 bloğa düşeriz, bonus 15 → 5'e iner.
4. **5-fold CV pahalı** — eğer zaman daralırsa 3-fold'a düşür ama README'de dürüstçe belirt.

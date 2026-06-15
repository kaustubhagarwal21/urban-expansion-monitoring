# Research Paper Explained — Line by Line

## Paper Title
**"Urban Expansion Monitoring Using Transfer Learning on Historical Satellite Imagery"**

**What it means in simple words:**
We built a system that uses satellite photos of Indian cities + AI models (originally trained on other image datasets) to track how cities are growing, predict future growth, and warn when construction happens in protected areas like forests or wetlands.

---

## ABSTRACT (The Summary)

**Sentence 1:** *"Uncontrolled urban expansion in Indian metropolitan regions accelerates encroachment on legally protected zones, yet existing studies rely on single-city, single-seed SVM/RF benchmarks without downstream forecasting or alerting."*

→ Indian cities are growing fast and eating into protected forests, wetlands, and coastal zones. But existing research only uses old-school ML methods (SVM, Random Forest) on ONE city at a time, without predicting future growth or generating alerts. Nobody has built a complete system for India.

**Sentence 2-3:** *"We propose an end-to-end framework integrating transfer-learning classification, SAR–optical fusion, self-supervised pre-training, uncertainty-aware Bi-LSTM forecasting, and regulatory encroachment alerting."*

→ We built a full pipeline with 5 components:
1. **Transfer-learning classification** — Take AI models pre-trained on ImageNet (millions of everyday photos) and fine-tune them on satellite images to classify land as Urban, Non-Urban, or Transition
2. **SAR-optical fusion** — Combine normal satellite photos (optical) with radar images (SAR) that can see through clouds
3. **Self-supervised pre-training** — Try training the model on unlabeled satellite images first (SimCLR method) before fine-tuning
4. **Bi-LSTM forecasting** — Use a recurrent neural network to predict how much cities will grow from 2024 to 2035, with uncertainty bands (confidence intervals)
5. **Regulatory alerting** — Automatically check if predicted growth zones overlap with protected areas and generate alerts

**Sentence 4:** *"A benchmark of 2,730 Sentinel-2 patches across Mumbai, Delhi NCR, and Bangalore..."*

→ We created a dataset of 2,730 small image tiles (256×256 pixels each) from the European Space Agency's Sentinel-2 satellite, covering 3 Indian cities. Each tile is labeled as Urban, Non-Urban, or Transition using ESA WorldCover 2021 (an existing global land cover map).

**Sentence 5:** *"Six models are trained with three-stage progressive fine-tuning and a combined CE–focal–Dice loss across three random seeds."*

→ We tested 6 models (SVM, Random Forest, MobileNetV3, EfficientNet-B0, Swin-Tiny, ResNet50). Deep learning models are trained in 3 stages (first only the top layer, then more layers, then everything). The loss function combines 3 different losses. Everything is run 3 times with different random seeds to prove results are not a fluke.

**Sentence 6:** *"ResNet50 achieves 97.5 ± 0.2% overall accuracy..."*

→ Our best model (ResNet50) correctly classifies 97.5% of patches, beating the best published Indian SVM result (91.01%) by 6 points.

**Sentence 7:** *"The first Leave-One-City-Out (LOCO) benchmark..."*

→ LOCO = Train on 2 cities, test on the 3rd. This tests if the model works on cities it has NEVER seen. Result: accuracy drops ~18% (from 97.5% to 79.1%). Interesting finding: Swin-Tiny (a Transformer model) generalizes BETTER to unseen cities than ResNet50 (a CNN), even though ResNet50 was more accurate on known cities. This "ranking reversal" is a key finding.

**Sentence 8:** *"The pipeline produces uncertainty-calibrated 2024–2035 forecasts and, in a simulation over seven cities, generates 55 encroachment alerts..."*

→ The system predicts city growth up to 2035 with confidence bands, and in a simulation, found 55 potential illegal constructions near protected areas like Sanjay Gandhi National Park.

---

## I. INTRODUCTION

### Problem Domain

**What it says in simple terms:**

India's cities are growing explosively — urban population will hit 600 million by 2031. This creates 3 problems:

1. **Illegal construction in protected areas** — Despite laws like CRZ (Coastal Regulation Zone), Forest Conservation Act, and Wetland Rules, people keep building in forests, wetlands, and near coasts. Examples: Sanjay Gandhi National Park (Mumbai), Bellandur wetland (Bangalore), Yamuna floodplain (Delhi).

2. **Nobody provides uncertainty in forecasts** — When city planners predict growth, they give one number with no error bars. This leads to bad infrastructure decisions (building too much or too little).

3. **Deep learning is underused for India** — AI models like ResNet50 get 97%+ accuracy on European satellite data (EuroSAT benchmark), but Indian studies still use basic SVM/RF methods that top out at 89-91%. Nobody has tested these modern models on Indian cities, tested cross-city generalization, or connected classification to forecasting and alerting.

**The gap we fill:** All the satellite data exists (Sentinel-2, Sentinel-1 SAR, Landsat via Google Earth Engine). What's missing is a complete pipeline that goes from raw satellite images → classification → prediction → alerts for India.

### Key Contributions (What's new in this paper)

1. **First multi-city Indian benchmark with LOCO** — 2,730 patches from 3 cities, 3-class labeling, tested with 3 random seeds for statistical rigor
2. **97.5% accuracy with progressive fine-tuning** — Beats published Indian SVM (91.01%) by 6 points, without any special domain-specific pre-training
3. **Novel Transition class** — We add a 3rd class (Transition) using a 100m buffer around urban edges. This captures the messy mixed-pixel zone where cities are actively expanding — most studies only use Urban/Non-Urban binary
4. **Uncertainty-aware forecasting** — Our Bi-LSTM gives 95% confidence intervals on predictions. No published Indian urban forecasting study does this
5. **End-to-end pipeline** — Classify → build time series → forecast → generate alerts. The alert system knows about 10 types of Indian protected zones (CRZ-I/II/III, Forest Reserve, Wetland, etc.) and 30+ specific named areas, and routes alerts to the correct government body (MoEFCC, CZMA, State Forest Dept)
6. **CNN-Transformer ranking reversal** — ResNet50 (CNN) wins on known cities, but Swin-Tiny (Transformer) wins on UNSEEN cities. This confirms a theory from computer vision research that Transformers handle distribution shifts better

---

## II. LITERATURE SURVEY

### What existing research says (simplified)

The literature survey table covers 16 key papers. Here's the landscape:

- **Backbone models (ResNet, EfficientNet, Swin, MobileNetV3):** These are pre-trained on ImageNet (14 million everyday photos) and then fine-tuned on satellite images. On European data (EuroSAT), they get 96-99% accuracy.

- **Indian urban studies:** The best published result using SVM on Indian Sentinel-2 data is 91.01% (Chamoli 2024). One study (Katpadi 2025) gets 98.21% but that's binary classification on a single region in Tamil Nadu — much easier task than ours.

- **Cross-city transfer:** Some studies (Li 2023, Wang 2023) test models across different cities, but only on European/Chinese cities. Nobody has done this for Indian cities.

- **Change detection:** Methods like GAS-Net reach F1=91.21% on LEVIR-CD (a change detection dataset), but they only DETECT change, they don't PREDICT future change or generate alerts.

- **CNN vs Transformer debate:** Research (Bai 2021) shows Transformers handle out-of-distribution data better than CNNs. Our results confirm this specifically for Indian satellite data.

- **Indian urban forecasting:** Existing studies (CA-Markov, MLP-Markov) predict future land use but give only point predictions with NO confidence intervals.

**THE GAP:** No one has combined, for Indian cities: multi-seed deep learning classification + cross-city testing + uncertainty-aware forecasting + regulatory alerting in one system.

---

## III. DATASET CHARACTERISTICS AND TOOLS

### Study Area

Three cities chosen to represent different growth patterns:
- **Mumbai** — Coastal megacity, squeezed between ocean and national park
- **Delhi NCR** — Spreads outward in all directions like a circle (radial sprawl)
- **Bangalore** — IT corridor city with scattered pockets of development

### Satellite Data Used

| Data | Resolution | What it captures | Used for |
|------|-----------|-----------------|----------|
| **Sentinel-2** | 10m per pixel | Visual + infrared bands (6 bands) | Main classification training (2017-2023) |
| **Sentinel-1 SAR** | 10m per pixel | Radar signals (sees through clouds) | Fusion experiment (2023 only) |
| **Landsat 5/7/8/9** | 30m per pixel | Historical imagery | Long-term trend analysis only (1990-2023) |

### Labels (Ground Truth)

- Labels come from **ESA WorldCover 2021** — a global 10m land cover map made by ESA
- This is "weak supervision" — the labels are automatically generated, NOT manually verified by humans
- Cross-checked against **Google Dynamic World** (another global land cover product)

### Three-Class System

| Class | What it means | How it's defined |
|-------|--------------|-----------------|
| **Urban** | Built-up area | WorldCover class 50 |
| **Non-Urban** | Everything else (forest, water, farmland) | All other WorldCover classes |
| **Transition** | The messy edge where cities are actively growing | 100m buffer zone around Urban boundaries (created using morphological dilation) |

The Transition class is our innovation — it captures the mixed pixels where urban meets rural, which is exactly where illegal encroachment happens.

### Dataset Size
- **2,730 patches** total (256×256 pixels each)
  - Mumbai: ~900 patches
  - Delhi NCR: ~1,200 patches
  - Bangalore: ~630 patches
- Class distribution: Urban 19.2%, Non-Urban 62.4%, Transition 18.4%
- 910 patches have matching SAR data
- Split: 70% train, 15% validation, 15% test

### Additional Data
- **LEVIR-CD** — 637 bi-temporal image pairs for change detection testing
- **Socio-economic features** for forecasting: Census population, GDP, Smart City funding, metro rail length, highway density, IT park counts, green cover percentage, etc.

### Hardware
- NVIDIA RTX 4070 Laptop GPU (8GB VRAM)
- All experiments run 3 times with seeds {42, 123, 7}

---

## IV. PROPOSED METHODOLOGY

### How the system works (step by step)

#### Step 1: Classification — What is each patch?

**Backbones** — We take 4 pre-trained deep learning models:
| Model | Parameters | Role |
|-------|-----------|------|
| ResNet50 | 11.3 million | Primary model (best accuracy) |
| EfficientNet-B0 | 5.3 million | Used for ablation experiments |
| Swin-Tiny | 29.8 million | Best at generalizing to new cities |
| MobileNetV3-Small | 3.4 million | Smallest, for edge/mobile deployment |

Plus 2 traditional ML baselines: SVM and Random Forest.

**FPN (Feature Pyramid Network)** — Takes features from 3 different scales (zoom levels) in the backbone and combines them. This helps the model see both fine details (individual buildings) and big patterns (neighborhoods).

**Progressive Fine-Tuning** — Instead of training the whole model at once, we do it in 3 stages:
- **Stage 1 (5 epochs):** Only train the top classification layer. Learning rate = 0.001. The backbone stays frozen (its weights don't change).
- **Stage 2 (5 epochs):** Unfreeze the last 2 blocks of the backbone. Learning rate = 0.0001 (slower, so we don't destroy pre-trained knowledge).
- **Stage 3 (5 epochs):** Unfreeze everything. Learning rate = 0.00001 (very slow, fine adjustments only).

This prevents "catastrophic forgetting" — where fine-tuning destroys the useful features the model learned from ImageNet.

#### Step 2: The Loss Function

```
Total Loss = 0.6 × CrossEntropy + 0.3 × FocalLoss(γ=2) + 0.1 × DiceLoss
```

**Why 3 losses combined?**
- **Cross-Entropy (CE):** Standard classification loss. Good general-purpose loss.
- **Focal Loss:** Pays MORE attention to hard-to-classify examples (the ones the model gets wrong). γ=2 means it strongly downweights easy examples.
- **Dice Loss:** Specifically helps with class imbalance. Makes sure the model doesn't ignore the minority Transition class.
- **Class weights [1.0, 1.0, 3.0]:** The Transition class gets 3× more importance because it has fewer samples.

#### Step 3: SAR-Optical Fusion (Pillar I)

**What:** Combine regular photos (optical) with radar images (SAR).
**Why:** SAR can see through clouds (important during India's monsoon season).
**How:** Two separate branches process optical (6 channels) and SAR (2 channels), then their features are concatenated and fed to the shared FPN.
**Result:** Didn't actually help (87.9% vs 96.7% optical-only) because SAR data had seasonal mismatch — optical was pre-monsoon but SAR was post-monsoon.

#### Step 4: SimCLR Self-Supervised Pre-Training (Pillar II)

**What:** Before fine-tuning on labeled data, first train the model on UNLABELED satellite patches using contrastive learning.
**How SimCLR works:** Take one image, create 2 different augmented versions (crop, color change, blur). Train the model to recognize that these 2 versions came from the SAME image while being different from all other images.
**Result:** Didn't help either (93.2% vs 96.9% with ImageNet). With enough labeled data (2,730 patches), ImageNet pre-training is already good enough. SimCLR would help if we had fewer than ~500 labeled patches.

#### Step 5: Bi-LSTM Forecasting (Pillar IV)

**What:** Predict how much each city will grow from 2024 to 2035.
**How:**
1. From classified satellite images, count urban pixels per city per year → gives urban area in km²
2. Combine with socio-economic features (population, GDP, policy events like Smart City Mission)
3. Feed this time series into a Bi-LSTM (Bidirectional Long Short-Term Memory) neural network
4. The LSTM looks at patterns both forward and backward in time
5. Multi-head attention (2 heads) helps the model focus on the most important time steps
6. **MC Dropout (Monte Carlo Dropout):** At prediction time, randomly drop neurons 50 times and take the average. The spread of these 50 predictions gives us 95% confidence intervals.

**Key point:** The LSTM's main value is NOT higher accuracy (Ridge regression actually beats it on raw numbers), but the fact that it provides UNCERTAINTY BANDS. No other Indian forecasting study does this.

#### Step 6: Alert Engine (Pillar V)

**What:** Automatically detect if predicted urban expansion would encroach on protected areas.
**How:**
1. The 3-head change detector classifies each patch for: (a) change or no-change, (b) severity level (NONE/LOW/MEDIUM/HIGH/CRITICAL), (c) type of encroachment
2. The system knows about 10 types of Indian protected zones: CRZ-I, CRZ-II, CRZ-III, Forest Reserve, Protected Forest, Wetland, Lake Buffer, River Floodplain, Green Belt, Western Ghats ESA
3. It has a database of 30+ specific named protected areas (Sanjay Gandhi NP, Pallikaranai marsh, Yamuna floodplain, etc.)
4. Alerts are automatically routed to the correct authority:
   - Forest encroachment → MoEFCC (Ministry of Environment)
   - Coastal encroachment → CZMA (Coastal Zone Management Authority)
   - Forest reserve encroachment → State Forest Department

**Important:** This alert system is simulation-based — it demonstrates the concept but hasn't been validated against real encroachment records.

#### The Full Pipeline (End to End)

```
Satellite Image (GeoTIFF)
    ↓
Slide 256×256 window across the image
    ↓
Classify each window as Urban/Non-Urban/Transition
    ↓
Count urban pixels per city per year → urban area in km²
    ↓
Feed time series + socio-economic data into Bi-LSTM
    ↓
Get 2024-2035 predictions WITH 95% confidence intervals
    ↓
Check predicted growth zones against protected area database
    ↓
Generate alerts + route to correct government authority
```

---

## V. RESULT ANALYSIS AND DISCUSSIONS

### Main Benchmark Results (Table 1)

| Model | Accuracy | What it means |
|-------|---------|---------------|
| SVM (raw pixels) | 89.2% | Basic SVM on flattened pixel values |
| SVM (improved) | 92.6% | SVM with spectral indices (NDVI, NDBI, etc.) + PCA + grid search. BEATS published Indian SVM (91.01%) |
| Random Forest | 88.2% | Basic RF |
| RF (improved) | 90.7% | RF with same feature engineering |
| MobileNetV3-Small | 91.5% | Smallest DL model, edge-deployable |
| EfficientNet-B0 | 93.4% | Efficient CNN |
| Swin-Tiny | 93.6% | Transformer model |
| **ResNet50** | **97.5%** | **Best overall — wins on every metric** |

**Key takeaways:**
1. ALL deep learning models beat ALL traditional ML models
2. Even our IMPROVED SVM (with fancy feature engineering) still loses to deep learning by 5 points
3. ResNet50 has the lowest variance (±0.2%) — most stable and reliable
4. Our improved SVM (92.6%) beats the best published Indian SVM (91.01%) — every single seed beats it

### Cross-City LOCO Results (Table 2)

Train on 2 cities, test on the 3rd. This is the "real-world" test — can the model work on a city it has never seen?

| Model | In-distribution accuracy | LOCO accuracy | Drop |
|-------|------------------------|---------------|------|
| EfficientNet-B0 | 93.4% | 76.8% | -16.6% |
| ResNet50 | 97.5% | 77.1% | **-20.4%** |
| **Swin-Tiny** | 93.6% | **79.1%** | **-14.5%** |

**The CNN-Transformer Ranking Reversal:**
- On KNOWN cities: ResNet50 (CNN) > Swin-Tiny (Transformer)
- On UNKNOWN cities: Swin-Tiny (Transformer) > ResNet50 (CNN)
- ResNet50 drops 20.4% but Swin-Tiny only drops 14.5%

**Why?** Transformers use self-attention, which learns abstract, transferable patterns (general "urban-ness"). CNNs use local filters that learn city-specific textures (Mumbai's coastline patterns, Delhi's road grid patterns). When you move to a new city, the CNN's city-specific knowledge becomes useless.

**The 18% domain gap:** The drop from 97.5% → 79.1% is ~18%. A 2025 systematic review of 89 Sentinel-2 studies independently found that accuracy drops 15-25% when going from benchmark to real deployment. Our 18% falls right in the middle — this validates our finding.

**Per-city difficulty:**
- Delhi NCR is the HARDEST held-out target (71.8-76.7%) — its gradual radial sprawl is hard to learn from Mumbai/Bangalore
- Bangalore is the EASIEST held-out target (80.4-83.9%) — but it's the HARDEST in-distribution city
- This separation between "locally hard" and "hard to transfer to" is itself a novel finding

### Ablation Study (Table 3a)

"Ablation" = remove one component at a time to see if it matters.

| Config | OA | What we learn |
|--------|-----|--------------|
| Full method (FPN + combined loss) | 95.6% | Baseline |
| Without FPN | 95.3% | FPN helps slightly (+0.3%) |
| CE loss only (no Focal/Dice) | 96.0% | CE-only is slightly BETTER on aggregate OA |

**But wait — CE-only scores higher?** Yes, on overall accuracy. But per-class analysis shows the combined loss improves Transition recall (the minority class). Since Transition is the class most relevant to detecting encroachment, we keep the combined loss despite slightly lower overall OA.

### SAR Fusion Results (Table 3b)

| Config | OA |
|--------|-----|
| Optical only | 96.7% |
| Optical + SAR | 87.9% |

**SAR fusion HURTS accuracy.** Why?
1. Only 33% of patches (910/2730) have matching SAR data
2. Seasonal mismatch: optical = pre-monsoon, SAR = post-monsoon
3. The SAR data adds noise rather than useful information

This is a **deliberately reported negative result** — it's honest and tells practitioners: "Don't blindly add SAR. Make sure temporal alignment is right first."

### SimCLR Results (Table 3b)

| Config | OA |
|--------|-----|
| ImageNet pre-training | 96.9% |
| SimCLR pre-training | 93.2% |

**ImageNet wins.** With 2,730 labeled patches, there's enough data for ImageNet transfer to work well. SimCLR's advantage only shows up when you have very few labels (<500).

### Efficiency (Table 3c)

| Model | Parameters | Inference time | Speed | GPU memory |
|-------|-----------|---------------|-------|------------|
| MobileNetV3-Small | 3.39M | 5.42ms | 184 patches/sec | 30.5 MB |
| EfficientNet-B0 | 5.25M | 7.14ms | 140 patches/sec | 52.9 MB |
| ResNet50 | 11.31M | 5.02ms | 199 patches/sec | 83.4 MB |
| Swin-Tiny | 29.83M | 10.98ms | 91 patches/sec | 176.3 MB |

**ResNet50 is actually the FASTEST** despite having more parameters than MobileNetV3 — this is because ResNet50's architecture is more GPU-friendly (simple convolutions parallelize well).

### Forecasting and Alerts

**Bi-LSTM Forecasting:**
- On clean socio-economic data: R² = 0.9564 (explains 95.6% of variance)
- On real satellite-derived data: R² = 0.559 (drops because real satellite estimates are noisy)
- Ridge regression baseline: R² = 0.9743 (actually beats LSTM on raw accuracy)
- **LSTM's value = uncertainty quantification**, not accuracy

**Siamese Change Detection:**
- On LEVIR-CD benchmark: F1 = 0.949, OA = 94.5%
- Competitive with published state-of-the-art

**Alert System (simulation):**
- 55 simulated alerts across 7 cities
- 4 CRITICAL, 28 HIGH, 2 MEDIUM, 21 LOW
- 11 violations near protected zones
- Critical alerts near Sanjay Gandhi National Park and Pallikaranai marsh
- Alerts routed to MoEFCC (federal) or state authorities

**GradCAM Analysis:**
- Shows WHERE the model is looking when making decisions
- Urban patches: model focuses on built-up structures
- Non-Urban patches: minimal, diffuse activation
- Transition patches: scattered attention at construction edges
- Confirms the model learned meaningful urban features

**Temporal Validation:**
- Applied 2021-trained model to 2019 and 2023 images without retraining
- Mumbai urban extent: 1,161 km² (2019) → 1,451 km² (2023) = +25% growth
- Consistent with known coastal construction patterns

---

## VI. CONCLUSION

**What we did:**
Built a complete system that goes from satellite images → land classification → growth prediction → encroachment alerts, specifically for Indian cities.

**Key results:**
- ResNet50: 97.5% accuracy (beats published Indian SVM by 6 points)
- First Indian LOCO benchmark: 18% cross-city drop
- CNN-Transformer reversal: Swin-Tiny generalizes better than ResNet50
- Uncertainty-calibrated forecasts for 2024-2035
- 55 simulated alerts with regulatory routing

**Limitations (honest):**
1. Only 3 cities (all Tier-1 metros) — doesn't cover smaller cities
2. Labels are from WorldCover (weak supervision, not manually verified)
3. Transition class uses automatic dilation (not human-drawn boundaries)
4. Forecasting uses calibrated socio-economic series
5. Alert system is simulation-based (not tested against real encroachment)
6. SAR fusion suffers from seasonal mismatch
7. Only 3 seeds for statistical testing (some differences may not be significant)

**Future work:**
- Expand to Tier-2/Tier-3 cities
- Validate alerts against real government encroachment records
- Fix SAR fusion with proper temporal alignment

---

## APPENDIX: Reproducibility

Everything needed to reproduce the results:
- Exact hardware (RTX 4070, 8GB VRAM)
- Exact software versions (PyTorch 2.0.1, timm 0.9.12, etc.)
- Exact seeds (42, 123, 7)
- Every hyperparameter (batch size, learning rates, loss weights, etc.)
- Data access instructions (GEE project name, file paths)
- Code will be released on GitHub upon acceptance

---

## GLOSSARY OF TECHNICAL TERMS

| Term | Simple Meaning |
|------|---------------|
| **Transfer Learning** | Take a model trained on one task (ImageNet photos) and adapt it for another (satellite images) |
| **Fine-tuning** | Adjusting a pre-trained model's weights on new data |
| **Progressive Fine-tuning** | Fine-tuning in stages: first top layer, then more, then everything |
| **Backbone** | The main feature extraction part of a neural network (ResNet50, Swin-Tiny, etc.) |
| **FPN (Feature Pyramid Network)** | Combines features from multiple scales to see both details and big patterns |
| **Cross-Entropy Loss** | Standard loss function for classification |
| **Focal Loss** | Variant of CE that focuses on hard examples |
| **Dice Loss** | Loss that handles class imbalance well |
| **SVM (Support Vector Machine)** | Traditional ML classifier that finds the best separating boundary |
| **Random Forest** | Ensemble of decision trees |
| **CNN (Convolutional Neural Network)** | Neural network that uses filters to detect patterns in images |
| **Transformer** | Architecture using self-attention instead of convolutions |
| **Self-attention** | Mechanism where each pixel looks at all other pixels to understand context |
| **LOCO (Leave-One-City-Out)** | Train on 2 cities, test on the 3rd — tests generalization |
| **OA (Overall Accuracy)** | Percentage of correctly classified patches |
| **F1 Score** | Harmonic mean of precision and recall — better than OA for imbalanced data |
| **mIoU (mean Intersection over Union)** | Measures overlap between predicted and true classes — strictest metric |
| **Sentinel-2** | ESA satellite, 10m resolution, optical (visual + infrared) |
| **Sentinel-1 SAR** | ESA satellite, 10m resolution, radar (sees through clouds) |
| **Landsat** | NASA/USGS satellite, 30m resolution, available since 1972 |
| **GEE (Google Earth Engine)** | Cloud platform for processing satellite imagery |
| **ESA WorldCover** | Global 10m land cover map from ESA |
| **Bi-LSTM** | Bidirectional LSTM — reads time series both forward and backward |
| **MC Dropout** | Monte Carlo Dropout — run prediction 50 times with random dropout to get uncertainty |
| **95% CI (Confidence Interval)** | Range where the true value falls 95% of the time |
| **SimCLR** | Self-supervised learning method using contrastive learning |
| **GradCAM** | Technique to visualize what part of an image the model focuses on |
| **Siamese Network** | Two identical networks that compare two images to detect changes |
| **LEVIR-CD** | Public dataset of before/after satellite image pairs for change detection |
| **Domain shift/gap** | Accuracy drop when a model is applied to data from a different source/location |
| **CRZ** | Coastal Regulation Zone — Indian law protecting coastal areas |
| **MoEFCC** | Ministry of Environment, Forest and Climate Change (India) |
| **CZMA** | Coastal Zone Management Authority |
| **NDVI/NDBI/NDWI** | Spectral indices derived from satellite bands — highlight vegetation/built-up/water |
| **PCA** | Principal Component Analysis — reduces data dimensions while keeping important info |
| **Weak supervision** | Labels generated automatically (not by human annotators) |
| **Ablation study** | Remove components one at a time to see which ones matter |
| **Statistical significance (p < 0.05)** | Less than 5% chance the difference is due to random luck |

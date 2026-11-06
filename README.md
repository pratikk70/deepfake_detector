# Granular Computing-Based Multi-Level Feature Analysis for Robust Deepfake Detection
**7th Semester Minor Project**

An end-to-end, interpretable deepfake detection framework that decomposes human faces into **three complementary granular levels** (Microscopic Pixel Noise Residuals, Mesoscopic LBP Texture Gradients, and Macroscopic Anatomical Facial Regions) fused dynamically through an **Adaptive Attention-Based Evidence Gating Network**.

---

## 🚀 Quick Start Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Dataset & Preprocessing
```bash
python src/dataset.py
```
*(Automatically ensures at least 60 Real and 60 Fake face samples exist in `data/real` and `data/fake`. If you have FaceForensics++ MP4/AVI videos, simply drop them into `data/real` or `data/fake` and the script extracts 128x128 face crops automatically).*

### 3. Verify Models & Granule Extraction
```bash
python src/granules_and_model.py
```

### 4. Train Models & Generate Presentation Charts
```bash
python src/train_and_eval.py
```
*(Trains both the Single-Level Baseline CNN and the 3-Branch Granular Network for 15 epochs, runs FGSM and PGD adversarial robustness sweeps, and generates publication-grade PNG charts and JSON metrics into `saved_models/`).*

### 5. Launch Interactive Streamlit Presentation Dashboard
```bash
streamlit run app.py
```

---

## 🏛️ Architecture Overview

```
                      +-----------------------------+
                      |   Input Face (128x128x3)    |
                      +--------------+--------------+
                                     |
           +-------------------------+-------------------------+
           |                         |                         |
           v                         v                         v
  +------------------+      +------------------+      +------------------+
  |    GRANULE 1     |      |    GRANULE 2     |      |    GRANULE 3     |
  |  (Fine-Grained)  |      |  (Medium-Grain)  |      |  (Coarse-Grain)  |
  |   Pixel Noise    |      |  Texture & Edge  |      | 4 Facial Regions |
  |    Residuals     |      |    (LBP/Sobel)   |      | (Eyes,Nose,Mouth,|
  |  (Laplacian HPF) |      | (Directional Mag)|      |    Cheek/Jaw)    |
  +--------+---------+      +--------+---------+      +--------+---------+
           |                         |                         |
           | f1 (32-dim)             | f2 (32-dim)             | f3 (32-dim)
           +-------------------------+-------------------------+
                                     |
                                     v
                      +------------------------------+
                      | Adaptive Evidence Gating     |
                      | Softmax([f1; f2; f3])        |
                      | Trust: [w_pixel, w_tex, w_reg|
                      +--------------+---------------+
                                     |
                                     v
                      +------------------------------+
                      |  Fused Evidence Classifier   |
                      |  Real vs. Fake Decision      |
                      +------------------------------+
```

---

## 📊 Live Presentation Dashboard Features (`app.py`)

1. **Tab 1: Multi-Granular Feature Inspector**
   - Displays 128x128 face input with annotated anatomical bounding boxes.
   - Visualizes the 3 granules side-by-side:
     - High-pass Pixel Noise Residual map (5x amplified).
     - Local Binary Patterns (LBP) skin texture map.
     - 4 anatomical facial region crops (Eyes, Nose, Mouth, Cheek/Jawline).
   - Real-time classification badge (`REAL` / `DEEPFAKE`) with confidence score and interactive horizontal bar chart showing the % trust contribution of each granule.

2. **Tab 2: Live Adversarial Attack Lab (FGSM & PGD)**
   - Interactive Epsilon Slider ($\epsilon \in [0.00, 0.08]$) for FGSM and PGD attacks.
   - Shows 3 images side-by-side: Clean Face, Amplified Adversarial Perturbation Noise, and Attacked Face.
   - Live side-by-side comparison proving that the Single-Level Baseline model fails while the Multi-Granular network stays robust.
   - Demonstrates the **Dynamic Trust Shift**: under adversarial noise, the gating layer automatically suppresses the corrupted Pixel Granule and shifts reliance to Texture and Anatomical cues!

3. **Tab 3: Evaluation Metrics & PPT Charts**
   - Displays test set Accuracy, Precision, Recall, and F1 scores.
   - Pre-rendered, 300 DPI figures ready for download and insertion into your project presentation slides (`confusion_matrices.png` and `robustness_curve.png`).

---

## 🎓 Viva Voce Quick Reference

- **Q: Why use Granular Computing instead of a deeper CNN like ResNet-50?**
  *A: Standard monolithic CNNs overfit to single frequency bands (e.g. high-frequency GAN noise). When attacked with FGSM noise or compressed, they collapse. Granular Computing enforces orthogonal feature extraction at micro (pixel), meso (texture), and macro (facial anatomy) scales.*
- **Q: How does the Adaptive Evidence Fusion layer defend against attacks?**
  *A: A gating network computes data-dependent Softmax trust weights. When adversarial perturbation corrupts the pixel branch, its activation pattern becomes abnormal, and the gating mechanism dynamically down-weights $w_{\text{pixel}}$ while increasing $w_{\text{texture}}$ and $w_{\text{region}}$.*
- **Q: Are the parameter counts comparable?**
  *A: Yes. The Baseline Monolithic Net has 25,921 parameters and the Proposed Granular Net has 23,924 parameters. The robustness gain is purely algorithmic, not from model capacity.*

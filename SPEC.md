# Granular Computing-Based Multi-Level Feature Analysis for Robust Deepfake Detection
**7th Semester Minor Project Specification & Viva Guide**

---

## 1. Project Overview & Motivation

Deepfake generation techniques (such as GANs, Diffusion Models, and Face-Swap autoencoders) have become increasingly sophisticated. Most existing deepfake detectors rely on single-granularity convolutional neural networks (CNNs) evaluating the entire face holistically. These monolithic networks suffer from two key vulnerabilities:
1. **Brittleness to Noise & Compression**: High-frequency GAN artifacts are easily destroyed by social media compression (e.g., H.264 compression, JPEG compression, Gaussian blur).
2. **Susceptibility to Adversarial Attacks**: Holistic models are easily fooled by small, imperceptible perturbations (e.g., FGSM, PGD) that disrupt global decision boundaries.

### Core Philosophy: Granular Computing (GrC)
Granular Computing is an information processing paradigm that decomposes complex problems into granules of varying sizes, scales, and abstractions (fine to coarse). By analyzing an image across **three distinct granular levels**, our system achieves:
- **Complementary Evidence Fusion**: Different manipulation signatures manifest at different scales.
- **Robustness**: If one granular level is corrupted or attacked (e.g., noise residual degraded by compression or adversarial noise), the other granules (texture and facial anatomy) maintain high predictive confidence.
- **Explainability**: Clear attribution indicating *which* granule (noise, texture, or anatomical region) contributed most to the fake/real classification.

---

## 2. Multi-Level Granular Architecture

```
                    +---------------------------+
                    | Input Face Image (224x224)|
                    +-------------+-------------+
                                  |
         +------------------------+------------------------+
         |                        |                        |
         v                        v                        v
+------------------+     +------------------+     +------------------+
|    GRANULE 1     |     |    GRANULE 2     |     |    GRANULE 3     |
|   (Fine-Grain)   |     |  (Medium-Grain)  |     |  (Coarse-Grain)  |
|   Pixel Noise    |     |  Texture & Edge  |     | 4 Facial Regions |
|    Residuals     |     |    (LBP/Sobel)   |     | (Eyes,Nose,Mouth,|
|                  |     |                  |     |    Cheek/Jaw)    |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         | Feature Vector f1      | Feature Vector f2      | Feature Vector f3
         +------------------------+------------------------+
                                  |
                                  v
                   +------------------------------+
                   | Adaptive Granular Fusion     |
                   | (Attention / Gating Network) |
                   +--------------+---------------+
                                  |
                                  v
                   +------------------------------+
                   |  Classification & Evidence   |
                   |       (Real vs. Fake)        |
                   +------------------------------+
```

---

### Granule 1: Fine-Grained (Pixel Noise Residuals)
- **Concept**: Forensic noise analysis at the microscopic pixel level.
- **Mechanism**:
  - Image smoothing filters (e.g., High-Pass / Median filter or Spatial Rich Model (SRM) kernels) are applied to suppress low-frequency semantic content.
  - The pixel noise residual is computed as:
    $$R(x, y) = I(x, y) - \text{Filter}(I(x, y))$$
- **What it captures**:
  - Micro-artifacts left by generative upsampling layers (checkerboard artifacts, deconvolution patterns).
  - Sensor PRNU (Photo-Response Non-Uniformity) discrepancies between spliced areas and background.
- **Viva Keypoint**: *"Granule 1 strips away semantic face appearance and retains only high-frequency noise residuals where GAN synthesis prints distinct spectral fingerprints."*

---

### Granule 2: Medium-Grained (Texture & Edge Patterns)
- **Concept**: Statistical and gradient texture analysis across mesoscopic patches.
- **Mechanism**:
  - **Local Binary Patterns (LBP)**: Encodes micro-texture by thresholding each pixel against its $P$ neighbors at radius $R$:
    $$LBP_{P, R}(x_c, y_c) = \sum_{p=0}^{P-1} s(i_p - i_c) 2^p, \quad s(z) = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}$$
  - **Sobel / Gradient Magnitude**: Detects unnatural sharpness or gradient discontinuity along boundaries.
  - **GLCM Statistics**: Contrast, correlation, energy, and homogeneity metrics of the texture map.
- **What it captures**:
  - Over-smoothed skin surfaces (a common artifact in autoencoder blending).
  - Unnatural edge transitions between face swaps and background skin.
- **Viva Keypoint**: *"Granule 2 analyzes structural micro-textures using LBP and Sobel gradients. Generative models struggle to replicate organic human pore patterns and subtle skin texture variations."*

---

### Granule 3: Coarse-Grained (4 Anatomical Facial Regions)
- **Concept**: Macroscopic, semantic decomposition based on human facial anatomy.
- **Mechanism**:
  - The aligned face is segmented into 4 key anatomical regions:
    1. **Eyes & Eyebrows**: Gaze consistency, pupil reflections, iris circularity.
    2. **Nose Bridge & Tip**: Lighting direction, specular highlights.
    3. **Mouth & Teeth**: Dentition realism, lip boundary artifacts, phoneme-viseme coherence.
    4. **Cheeks & Jawline**: Boundary blending seams, color warping, face mask contours.
- **What it captures**:
  - Regional inconsistencies (e.g., real eyes but warped mouth, or blending seam along the jawline).
  - Lighting and geometric asymmetry across anatomical landmarks.
- **Viva Keypoint**: *"Granule 3 decomposes the face into 4 anatomical zones: Eyes, Nose, Mouth, and Cheek/Jaw. This regional granularity prevents local blending artifacts from being averaged out by global pooling."*

---

## 3. Adaptive Granular Fusion Classifier

Rather than simple concatenation, the model employs **Adaptive Attention-Based Evidence Fusion**:

1. **Feature Extraction**:
   - Each granule produces a latent representation:
     - $f_1 \in \mathbb{R}^{D}$ (Noise Residual Branch)
     - $f_2 \in \mathbb{R}^{D}$ (Texture Branch)
     - $f_3 \in \mathbb{R}^{D}$ (Regional Multi-Crop Branch)

2. **Dynamic Attention Gating**:
   - A lightweight gating network computes adaptive confidence weights $\alpha_1, \alpha_2, \alpha_3$ such that $\sum_{i=1}^3 \alpha_i = 1$:
     $$\mathbf{\alpha} = \text{Softmax}(W_g [f_1; f_2; f_3] + b_g)$$
   - The fused feature vector is computed as:
     $$f_{\text{fused}} = \sum_{i=1}^3 \alpha_i \cdot f_i$$

3. **Classification & Evidence Output**:
   - Final classification: $\hat{y} = \sigma(W_c f_{\text{fused}} + b_c) \in [0, 1]$
   - **Granular Contribution Breakdown**: The model outputs $[\alpha_1, \alpha_2, \alpha_3]$, providing human-interpretable rationale for the prediction.

---

## 4. Adversarial Attack & Robustness Testing

To evaluate real-world resilience, the detector is tested under intentional adversarial perturbations:
- **FGSM (Fast Gradient Sign Method)**:
  $$x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(\theta, x, y))$$
- **PGD (Projected Gradient Descent)**:
  Multi-step iterative perturbation projected within an $\epsilon$-ball:
  $$x^{t+1} = \Pi_{x + \mathcal{S}} \left( x^t + \alpha \cdot \text{sign}(\nabla_{x^t} \mathcal{L}(\theta, x^t, y)) \right)$$
- **Granular Defense Hypothesis**: When adversarial noise disrupts the high-frequency pixel granule ($f_1$), the adaptive gating mechanism down-weights $\alpha_1$ and prioritizes structural and regional granules ($\alpha_2, \alpha_3$), maintaining robustness.

---

## 5. Technology Stack Summary

| Component | Library / Framework | Purpose |
| :--- | :--- | :--- |
| Deep Learning | PyTorch, torchvision | 3-Branch Granular Network, Attention Fusion, FGSM/PGD attacks |
| Vision & Preprocessing | OpenCV (`cv2`), NumPy | Video decoding, face crop, region slicing, Sobel filters |
| Texture Analysis | `scikit-image` | LBP (Local Binary Patterns), GLCM texture maps |
| Evaluation Metrics | `scikit-learn`, Matplotlib | Confusion matrices, ROC curves, accuracy vs $\epsilon$ plots |
| Web Application | Streamlit | Interactive inference dashboard with live webcam/video/image upload |
| Image IO | Pillow (`PIL`) | Image handling and format conversion |

---

## 6. Directory Structure

```
granular_detector/
├── SPEC.md                  # Project specification & Viva guide (this file)
├── requirements.txt         # Python dependencies
├── data/                    # Dataset directory
│   ├── real/                # Real face images / frames
│   └── fake/                # Deepfake face images / frames
├── saved_models/            # Checkpoints (.pt / .pth)
└── src/                     # Core Python modules
    ├── __init__.py
    ├── preprocessing.py     # Face detection, noise extraction, LBP, 4-region slicing
    ├── dataset.py           # PyTorch Dataset & DataLoader with multi-granule transforms
    ├── models.py            # 3-Branch Granular Architecture & Attention Fusion Head
    ├── attacks.py           # FGSM and PGD adversarial attack generators
    ├── train.py             # Training loop, validation, and checkpointing
    ├── evaluate.py          # Metrics, confusion matrix, ROC-AUC, attack evaluation
    └── app.py               # Interactive Streamlit dashboard
```

---

## 7. Viva Voce Quick Cheat Sheet

- **Q: Why call it "Granular Computing"?**
  *A: Granular Computing is a paradigm where information is structured into granules of different levels of abstraction. We split face analysis into microscopic pixel noise, mesoscopic texture patterns, and macroscopic facial regions.*
- **Q: What is the advantage over a single ResNet/EfficientNet?**
  *A: Standard CNNs overfit to specific artifact types and easily break when attacked or compressed. Granular decomposition forces the model to capture orthogonal, complementary cues, so failure in one domain does not collapse the entire detector.*
- **Q: How does the Adaptive Fusion work?**
  *A: Instead of static averaging, an attention network learns data-dependent weights for each branch. If noise residuals are corrupted by noise or attacks, the network automatically shifts weight to texture and regional branches.*
- **Q: How do you segment the 4 facial regions?**
  *A: Using bounding ratios normalized to the aligned face: Upper-middle for Eyes, Center for Nose, Lower-center for Mouth, and Lateral/Perimeter boundaries for Cheek/Jawline.*

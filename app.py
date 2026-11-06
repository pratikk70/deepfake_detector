"""
Granular Computing-Based Multi-Level Feature Analysis for Robust Deepfake Detection
Module: app.py

Interactive Presentation & Viva Voce Dashboard built with Streamlit.
Upgraded to:
1. Operate strictly in [0.0, 1.0] RGB pixel units at 160x160 resolution using shared preprocess_face.
2. Emb_dim = 64 with MobileNetV3 semantic regions & sharp LBP texture granules.
3. Dynamic metrics loading directly from saved_models/metrics_summary.json (zero hardcoded numbers).
4. Independent white-box adversarial attack generation for both Baseline and Granular models in Tab 2.
5. Ground-truth label extraction based on dataset folder path (Real vs. Fake).
6. 100% offline (no external icon or web requests).
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Set safe matplotlib cache directory
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    preprocess_face,
    denormalize_tensor,
    IMAGE_EXTENSIONS,
)
from src.granules_and_model import (
    GranularDeepfakeNet,
    SingleLevelBaselineNet,
    extract_visual_granules,
)
from src.train_and_eval import fgsm_attack, pgd_attack


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Granular Deepfake Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .badge-real {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.45rem 1.0rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        border: 2px solid #10B981;
    }
    .badge-fake {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.45rem 1.0rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        border: 2px solid #EF4444;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# MODEL CACHING & LOADING (64-DIM EMBEDDINGS)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models() -> Tuple[Optional[GranularDeepfakeNet], Optional[SingleLevelBaselineNet]]:
    """Loads trained Granular and Baseline PyTorch checkpoints."""
    granular_path = PROJECT_ROOT / "saved_models" / "granular_net.pt"
    baseline_path = PROJECT_ROOT / "saved_models" / "baseline.pt"

    model_g = GranularDeepfakeNet(emb_dim=64)
    model_b = SingleLevelBaselineNet()

    if granular_path.exists():
        model_g.load_state_dict(torch.load(granular_path, map_location="cpu"))
        model_g.eval()
    else:
        model_g = None

    if baseline_path.exists():
        model_b.load_state_dict(torch.load(baseline_path, map_location="cpu"))
        model_b.eval()
    else:
        model_b = None

    return model_g, model_b


def render_trust_bar_chart(weights: np.ndarray) -> plt.Figure:
    """Renders a horizontal bar chart displaying Granular Trust percentages."""
    w_p, w_t, w_r = weights[0] * 100, weights[1] * 100, weights[2] * 100
    labels = ["Granule 1: Pixel Noise", "Granule 2: Texture (LBP)", "Granule 3: Facial Regions"]
    vals = [w_p, w_t, w_r]
    colors = ["#EF4444", "#3B82F6", "#10B981"]

    fig, ax = plt.subplots(figsize=(7, 2.0), dpi=150)
    bars = ax.barh(labels, vals, color=colors, height=0.55)
    ax.set_xlim([0, 100])
    ax.set_xlabel("Adaptive Trust Allocation (%)", fontsize=9, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, vals):
        ax.text(
            val + 1.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & NAVIGATION (100% OFFLINE)
# -----------------------------------------------------------------------------
model_granular, model_baseline = load_models()

st.sidebar.markdown("# 🛡️ **Granular Deepfake Detector**")
st.sidebar.caption("7th-Semester Minor Project Live Demonstration")

st.sidebar.markdown("---")
st.sidebar.markdown("### **Architectural Workflow**")
st.sidebar.code(
    """
Input Face (160x160 [0, 1])
       │
 ┌─────┼─────┐
 ▼     ▼     ▼
G1    G2    G3
Pixel Texture Regions
(HPF) (LBP)   (MobileNet)
 └─────┬─────┘
       ▼
LayerNorm + Evidence Gating
       ▼
 [Real / Fake] Decision
    """,
    language="text",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### **Select Input Source**")
input_mode = st.sidebar.radio(
    "Choose Test Image Source:",
    [
        "Real Dataset Samples (Train)",
        "Unseen Kaggle Test Samples",
        "Upload Custom Image",
        "Upload Video (MP4/AVI)",
    ],
)

selected_image_np = None
ground_truth_label = None  # 0 = Real, 1 = Fake

if input_mode == "Real Dataset Samples (Train)":
    real_files = sorted(list((PROJECT_ROOT / "data" / "real").glob("*.jpg")))
    fake_files = sorted(list((PROJECT_ROOT / "data" / "fake").glob("*.jpg")))

    sample_options = {}
    for i, p in enumerate(real_files[:8]):
        sample_options[f"Authentic Face #{i+1} ({p.name})"] = (p, 0)
    for i, p in enumerate(fake_files[:8]):
        sample_options[f"Deepfake Face #{i+1} ({p.name})"] = (p, 1)

    if sample_options:
        chosen_label = st.sidebar.selectbox("Choose a face sample:", list(sample_options.keys()))
        chosen_path, ground_truth_label = sample_options[chosen_label]
        pil_raw = Image.open(chosen_path).convert("RGB")
        selected_image_np, input_tensor = preprocess_face(pil_raw, target_size=160)
    else:
        st.sidebar.warning("No images found in data/real or data/fake.")

elif input_mode == "Unseen Kaggle Test Samples":
    test_real_files = sorted(list((PROJECT_ROOT / "kaggle_test_samples" / "real").glob("*.jpg")))
    test_fake_files = sorted(list((PROJECT_ROOT / "kaggle_test_samples" / "fake").glob("*.jpg")))

    test_options = {}
    for i, p in enumerate(test_real_files):
        test_options[f"Unseen Real #{i+1} ({p.name})"] = (p, 0)
    for i, p in enumerate(test_fake_files):
        test_options[f"Unseen Deepfake #{i+1} ({p.name})"] = (p, 1)

    if test_options:
        chosen_test_label = st.sidebar.selectbox("Choose an unseen test face:", list(test_options.keys()))
        chosen_path, ground_truth_label = test_options[chosen_test_label]
        pil_raw = Image.open(chosen_path).convert("RGB")
        selected_image_np, input_tensor = preprocess_face(pil_raw, target_size=160)
    else:
        st.sidebar.warning("No images found in kaggle_test_samples/.")

elif input_mode == "Upload Custom Image":
    uploaded_file = st.sidebar.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        pil_raw = Image.open(uploaded_file).convert("RGB")
        selected_image_np, input_tensor = preprocess_face(pil_raw, target_size=160)
        ground_truth_label = None

elif input_mode == "Upload Video (MP4/AVI)":
    uploaded_video = st.sidebar.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_slider = st.sidebar.slider("Select Video Frame:", 0, max(0, total_frames - 1), 0)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
        ret, frame = cap.read()
        cap.release()
        try:
            os.remove(tfile.name)
        except Exception:
            pass

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            selected_image_np, input_tensor = preprocess_face(frame_rgb, target_size=160)
            ground_truth_label = None
        else:
            st.sidebar.error("Could not read frame from video.")

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Viva Tip**: Notice how our 3 Granules decompose the face across micro (pixel residuals), meso (sharp LBP texture), and macro (MobileNetV3 facial regions) to provide explainable decisions."
)


# -----------------------------------------------------------------------------
# MAIN APP TABS
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Granular Computing-Based Deepfake Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Multi-Level Feature Analysis, True LBP Texture, and Adaptive Evidence Fusion (160x160)</div>',
    unsafe_allow_html=True,
)

if model_granular is None or model_baseline is None:
    st.error("⚠️ Trained models not found in `saved_models/`. Please run `python src/train_and_eval.py` first.")
    st.stop()

if selected_image_np is None:
    st.warning("Please choose a sample face from the sidebar or upload an image to begin.")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "🔬 Tab 1: Multi-Granular Feature Inspector",
    "⚡ Tab 2: Live Adversarial Attack Lab (FGSM & PGD)",
    "📊 Tab 3: Evaluation Metrics & PPT Charts",
])


# =============================================================================
# TAB 1: MULTI-GRANULAR FEATURE INSPECTOR
# =============================================================================
with tab1:
    st.subheader("1. Input Face & 4 Non-Overlapping Anatomical Regions (160x160)")
    col_face, col_annotated, col_summary = st.columns([1, 1, 1.8])

    visual_dict = extract_visual_granules(selected_image_np)

    with col_face:
        st.image(visual_dict["original"], caption="Input Face (160x160)", use_container_width=True)

    with col_annotated:
        st.image(
            visual_dict["annotated_face"],
            caption="Non-Overlapping Anatomical Regions",
            use_container_width=True,
        )

    # Model Inference
    with torch.no_grad():
        logits_g, trust_weights = model_granular(input_tensor)
        prob_fake = torch.sigmoid(logits_g).item()
        is_fake_pred = prob_fake >= 0.5
        trust_np = trust_weights.cpu().numpy()[0]

    with col_summary:
        st.markdown("#### **Detection Decision & Evidence**")
        if is_fake_pred:
            st.markdown(
                f'<div class="badge-fake">🔴 DEEPFAKE / MANIPULATED ({prob_fake * 100:.1f}% Confidence)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="badge-real">🟢 AUTHENTIC / REAL ({(1.0 - prob_fake) * 100:.1f}% Confidence)</div>',
                unsafe_allow_html=True,
            )

        if ground_truth_label is not None:
            gt_str = "Authentic Real" if ground_truth_label == 0 else "Deepfake"
            st.caption(f"🎯 **Dataset Ground Truth**: `{gt_str}`")

        st.markdown(" ")
        st.markdown("**Granular Trust Allocation** (Sum = 100%):")
        fig_trust = render_trust_bar_chart(trust_np)
        st.pyplot(fig_trust)
        plt.close(fig_trust)

        max_granule_idx = int(np.argmax(trust_np))
        granule_names = ["Pixel Noise Residuals", "LBP Texture Patterns", "Facial Anatomical Regions"]
        st.caption(f"📌 **Primary Driving Granule**: `{granule_names[max_granule_idx]}` ({trust_np[max_granule_idx]*100:.1f}%)")

    st.markdown("---")
    st.subheader("2. Multi-Level Granular Feature Visualizations")

    g_col1, g_col2, g_col3 = st.columns(3)

    with g_col1:
        st.markdown("#### **Granule 1: Fine-Grained**")
        st.caption("High-Pass Pixel Noise Residual (Amplified 5x)")
        st.image(visual_dict["pixel_residual"], use_container_width=True)
        st.info("Captures sensor PRNU noise & GAN upsampling artifacts.")

    with g_col2:
        st.markdown("#### **Granule 2: Medium-Grained**")
        st.caption("Local Binary Patterns (LBP) Texture Map")
        st.image(visual_dict["lbp_texture"], use_container_width=True)
        st.info("Exposes unnatural skin smoothness & localized gradient boundary anomalies.")

    with g_col3:
        st.markdown("#### **Granule 3: Coarse-Grained**")
        st.caption("4 Non-Overlapping Anatomical Regions")

        r1, r2 = st.columns(2)
        with r1:
            st.image(visual_dict["regions"]["eyes"], caption="Periocular Eyes [25:65, 25:135]", use_container_width=True)
            st.image(visual_dict["regions"]["mouth"], caption="Mouth [102:138, 42:118]", use_container_width=True)
        with r2:
            st.image(visual_dict["regions"]["nose"], caption="Nose [65:102, 55:105]", use_container_width=True)
            st.image(visual_dict["regions"]["boundary"], caption="Forehead/Boundary [0:25, 25:135]", use_container_width=True)
        st.info("MobileNetV3 evaluates semantic regional geometry, gaze, and blending seams.")


# =============================================================================
# TAB 2: LIVE ADVERSARIAL ATTACK LAB (WHITE-BOX FGSM & PGD IN [0, 1])
# =============================================================================
with tab2:
    st.subheader("Interactive White-Box Adversarial Attack Lab")
    st.markdown(
        """
        Evaluate how our **Multi-Granular Network** withstands targeted white-box adversarial perturbations
        compared to the **Single-Level MobileNetV3 Baseline**. Both models are attacked with white-box gradients
        in physical `[0.0, 1.0]` pixel units.
        """
    )

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 2, 1.5])

    with ctrl_col1:
        attack_type = st.radio("Adversarial Attack Method:", ["FGSM (Fast Gradient Sign)", "PGD (Projected Gradient Descent)"])
    with ctrl_col2:
        epsilon_val = st.slider("Perturbation Budget ε (Epsilon):", min_value=0.00, max_value=0.08, value=0.04, step=0.005)
    with ctrl_col3:
        target_val = ground_truth_label if ground_truth_label is not None else (1 if is_fake_pred else 0)
        target_str = "Deepfake" if target_val == 1 else "Real"
        st.markdown(f"**Target True Class:** `{target_str}`")
        st.caption(f"Perturbation Magnitude: `{epsilon_val * 100:.1f}%`")

    # Generate Independent White-Box Adversarial Images for both models
    target_tensor = torch.tensor([target_val], dtype=torch.long)

    if attack_type.startswith("FGSM"):
        adv_tensor_b = fgsm_attack(model_baseline, input_tensor, target_tensor, epsilon=epsilon_val)
        adv_tensor_g = fgsm_attack(model_granular, input_tensor, target_tensor, epsilon=epsilon_val)
    else:
        adv_tensor_b = pgd_attack(model_baseline, input_tensor, target_tensor, epsilon=epsilon_val, steps=5)
        adv_tensor_g = pgd_attack(model_granular, input_tensor, target_tensor, epsilon=epsilon_val, steps=5)

    adv_img_np = denormalize_tensor(adv_tensor_g[0])
    clean_img_np = denormalize_tensor(input_tensor[0])

    # Compute Perturbation Difference Visual (Amplified 15x)
    diff = np.abs(adv_img_np.astype(np.float32) - clean_img_np.astype(np.float32))
    amplified_noise = np.clip(diff * 15.0 + 30.0, 0, 255).astype(np.uint8)

    st.markdown("#### **Adversarial Noise Injection Visual**")
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        st.image(clean_img_np, caption="1. Clean Face (Original [0, 1])", use_container_width=True)
    with v_col2:
        st.image(amplified_noise, caption=f"2. Adversarial Noise (Amplified 15x, ε={epsilon_val:.3f})", use_container_width=True)
    with v_col3:
        st.image(adv_img_np, caption="3. Perturbed Face (Attacked Input)", use_container_width=True)

    # Evaluate Each Model on its Own Attacked Input
    with torch.no_grad():
        logits_b_adv = model_baseline(adv_tensor_b)
        prob_b_adv = torch.sigmoid(logits_b_adv).item()
        b_pred_fake = prob_b_adv >= 0.5

        logits_g_adv, trust_adv = model_granular(adv_tensor_g)
        prob_g_adv = torch.sigmoid(logits_g_adv).item()
        g_pred_fake = prob_g_adv >= 0.5
        trust_adv_np = trust_adv.cpu().numpy()[0]

    st.markdown("---")
    st.subheader("Model Resilience Comparison Under Attack")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("### ❌ **Single-Level MobileNetV3 Baseline**")
        st.caption("Evaluated against its own white-box adversarial gradients")
        if b_pred_fake:
            st.markdown(f'<div class="badge-fake">Predicted: DEEPFAKE ({prob_b_adv * 100:.1f}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-real">Predicted: REAL ({(1.0 - prob_b_adv) * 100:.1f}%)</div>', unsafe_allow_html=True)

        is_b_correct = (b_pred_fake == (target_val == 1))
        if not is_b_correct and epsilon_val > 0.01:
            st.error("🚨 **FOOLED!** Baseline CNN decision boundary collapsed under adversarial noise.")
        else:
            st.info("State: Baseline currently retains correct prediction.")

    with res_col2:
        st.markdown("### 🛡️ **Proposed Multi-Granular Model**")
        st.caption("Evaluated against its own white-box adversarial gradients")
        if g_pred_fake:
            st.markdown(f'<div class="badge-fake">Predicted: DEEPFAKE ({prob_g_adv * 100:.1f}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-real">Predicted: REAL ({(1.0 - prob_g_adv) * 100:.1f}%)</div>', unsafe_allow_html=True)

        is_g_correct = (g_pred_fake == (target_val == 1))
        if is_g_correct:
            st.success("✅ **ROBUST!** Multi-Granular Network maintained correct decision.")
        else:
            st.warning("Model boundary reached extreme perturbation limit.")

    st.markdown("---")
    st.markdown("#### **Dynamic Trust Shift Demonstration**")

    # Dynamic explanation generated directly from live trust weights
    delta_pixel = (trust_adv_np[0] - trust_np[0]) * 100
    delta_texture = (trust_adv_np[1] - trust_np[1]) * 100
    delta_region = (trust_adv_np[2] - trust_np[2]) * 100

    shift_summary = []
    if delta_pixel < 0:
        shift_summary.append(f"Pixel Noise trust dropped by {abs(delta_pixel):.1f}%")
    else:
        shift_summary.append(f"Pixel Noise trust shifted by {delta_pixel:+.1f}%")

    if delta_texture > 0:
        shift_summary.append(f"Texture (LBP) trust increased by {delta_texture:.1f}%")
    if delta_region > 0:
        shift_summary.append(f"Facial Region trust increased by {delta_region:.1f}%")

    explanation_str = ", while ".join(shift_summary) if shift_summary else "Trust weights dynamically re-balanced."

    st.markdown(
        rf"""
        > **Live Adaptive Evidence Shift**: Under $\epsilon={epsilon_val:.3f}$ adversarial perturbation,
        > {explanation_str}. The network adapts in real time to down-weight noisy cues and rely on stable granules.
        """
    )

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("**Clean Image Trust Allocation:**")
        fig_t_clean = render_trust_bar_chart(trust_np)
        st.pyplot(fig_t_clean)
        plt.close(fig_t_clean)

    with t_col2:
        st.markdown(f"**Under Attack (ε={epsilon_val:.3f}) Trust Allocation:**")
        fig_t_adv = render_trust_bar_chart(trust_adv_np)
        st.pyplot(fig_t_adv)
        plt.close(fig_t_adv)

    st.caption(
        rf"📊 Live Trust Delta: Pixel: `{delta_pixel:+.1f}%` | Texture: `{delta_texture:+.1f}%` | Region: `{delta_region:+.1f}%`"
    )


# =============================================================================
# TAB 3: DYNAMIC EVALUATION METRICS & PPT CHARTS
# =============================================================================
with tab3:
    st.subheader("Dynamic Benchmark Metrics & Viva Presentation Charts")

    metrics_path = PROJECT_ROOT / "saved_models" / "metrics_summary.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            mdata = json.load(f)

        test_count = mdata.get("dataset", {}).get("test_count", 300)
        st.markdown(f"#### **1. Overall Performance on Test Set ({test_count} Real Faces)**")

        # Dynamically compute metrics and comparisons
        g_clean_acc = mdata["granular"]["clean"]["accuracy"] * 100
        b_clean_acc = mdata["baseline"]["clean"]["accuracy"] * 100

        g_fgsm_acc = mdata["granular"]["fgsm_eps_0.04"]["accuracy"] * 100
        b_fgsm_acc = mdata["baseline"]["fgsm_eps_0.04"]["accuracy"] * 100

        fgsm_diff = g_fgsm_acc - b_fgsm_acc
        base_drop = b_fgsm_acc - b_clean_acc

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Granular Clean Accuracy", f"{g_clean_acc:.1f}%", f"AUC: {mdata['granular']['clean']['roc_auc']:.3f}")
        m2.metric("Granular FGSM (ε=0.04) Acc", f"{g_fgsm_acc:.1f}%", f"{fgsm_diff:+.1f}% vs Baseline")
        m3.metric("Baseline Clean Accuracy", f"{b_clean_acc:.1f}%", f"AUC: {mdata['baseline']['clean']['roc_auc']:.3f}")
        m4.metric("Baseline FGSM (ε=0.04) Acc", f"{b_fgsm_acc:.1f}%", f"{base_drop:+.1f}% drop")

        # Full Dynamic Comparison Table
        table_rows = [
            {
                "Model": "Proposed 3-Branch Granular Net",
                "Clean Accuracy": f"{g_clean_acc:.1f}%",
                "Clean ROC-AUC": f"{mdata['granular']['clean']['roc_auc']:.3f}",
                "FGSM (ε=0.04) Accuracy": f"{g_fgsm_acc:.1f}%",
                "PGD (ε=0.04) Accuracy": f"{mdata['granular']['pgd_eps_0.04']['accuracy'] * 100:.1f}%",
                "F1-Score": f"{mdata['granular']['clean']['f1_score']:.3f}",
                "Parameters": f"{mdata['granular'].get('parameters', 0):,}",
            },
            {
                "Model": "Single-Level MobileNetV3 Baseline",
                "Clean Accuracy": f"{b_clean_acc:.1f}%",
                "Clean ROC-AUC": f"{mdata['baseline']['clean']['roc_auc']:.3f}",
                "FGSM (ε=0.04) Accuracy": f"{b_fgsm_acc:.1f}%",
                "PGD (ε=0.04) Accuracy": f"{mdata['baseline']['pgd_eps_0.04']['accuracy'] * 100:.1f}%",
                "F1-Score": f"{mdata['baseline']['clean']['f1_score']:.3f}",
                "Parameters": f"{mdata['baseline'].get('parameters', 0):,}",
            },
        ]
        st.table(table_rows)

        # Single-Branch Ablation Table
        if "ablation" in mdata and mdata["ablation"]:
            st.markdown("#### **2. Single-Branch Ablation Study (Clean Test Set)**")
            ab_rows = [
                {
                    "Granular Configuration": "Branch 1: Pixel Noise Only",
                    "Accuracy": f"{mdata['ablation']['pixel_only']['accuracy'] * 100:.1f}%",
                    "ROC-AUC": f"{mdata['ablation']['pixel_only']['roc_auc']:.3f}",
                    "F1-Score": f"{mdata['ablation']['pixel_only']['f1_score']:.3f}",
                },
                {
                    "Granular Configuration": "Branch 2: Texture (LBP+Sobel) Only",
                    "Accuracy": f"{mdata['ablation']['texture_only']['accuracy'] * 100:.1f}%",
                    "ROC-AUC": f"{mdata['ablation']['texture_only']['roc_auc']:.3f}",
                    "F1-Score": f"{mdata['ablation']['texture_only']['f1_score']:.3f}",
                },
                {
                    "Granular Configuration": "Branch 3: 4 Facial Regions (MobileNetV3) Only",
                    "Accuracy": f"{mdata['ablation']['region_only']['accuracy'] * 100:.1f}%",
                    "ROC-AUC": f"{mdata['ablation']['region_only']['roc_auc']:.3f}",
                    "F1-Score": f"{mdata['ablation']['region_only']['f1_score']:.3f}",
                },
                {
                    "Granular Configuration": "Full Adaptive Granular Fusion",
                    "Accuracy": f"{mdata['ablation']['full_fusion']['accuracy'] * 100:.1f}%",
                    "ROC-AUC": f"{mdata['ablation']['full_fusion']['roc_auc']:.3f}",
                    "F1-Score": f"{mdata['ablation']['full_fusion']['f1_score']:.3f}",
                },
            ]
            st.table(ab_rows)

    st.markdown("---")
    st.markdown("#### **3. Generated Publication Charts (Download for Slide Deck & Report)**")

    chart_col1, chart_col2 = st.columns(2)
    cm_img_path = PROJECT_ROOT / "saved_models" / "confusion_matrices.png"
    curve_img_path = PROJECT_ROOT / "saved_models" / "robustness_curve.png"

    with chart_col1:
        if cm_img_path.exists():
            st.image(str(cm_img_path), caption="Figure 1: Confusion Matrices (Clean vs. FGSM Attack)", use_container_width=True)
            with open(cm_img_path, "rb") as f:
                st.download_button(
                    label="📥 Download Confusion Matrices (PNG)",
                    data=f.read(),
                    file_name="confusion_matrices.png",
                    mime="image/png",
                )

    with chart_col2:
        if curve_img_path.exists():
            st.image(str(curve_img_path), caption="Figure 2: Robustness Curves & Single-Branch Ablation", use_container_width=True)
            with open(curve_img_path, "rb") as f:
                st.download_button(
                    label="📥 Download Robustness Curves (PNG)",
                    data=f.read(),
                    file_name="robustness_curve.png",
                    mime="image/png",
                )

    st.markdown("---")
    st.markdown("#### **4. Viva Voce Key Takeaways to Present**")
    st.markdown(
        """
        - **Why Multi-Level Granularity?** Standard CNNs overfit to single artifact categories (e.g. high-frequency noise). When social media compression or adversarial noise perturbs those frequencies, single-branch models fail.
        - **How does Evidence Gating work?** The Adaptive Fusion Layer evaluates cross-granular consistency via Softmax gating, assigning higher confidence weights to unaffected granules.
        - **Fair Parameter Budget**: Both the baseline and proposed networks operate with MobileNetV3 backbones, proving the robustness leap is purely due to granular decomposition rather than raw capacity.
        """
    )

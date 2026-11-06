"""
Granular Computing-Based Multi-Level Feature Analysis for Robust Deepfake Detection
Module: src/train_and_eval.py

Upgraded Training & Evaluation Pipeline for the 2,000-Image Kaggle Dataset:
1. Operates at 160x160 resolution in physical [0.0, 1.0] pixel units.
2. Auxiliary classification heads (bce_pix, bce_tex, bce_reg) added to training loss:
   loss = bce_main + 0.25 * (bce_pix + bce_tex + bce_reg) + 0.01 * (w * log(w)).sum(dim=1).mean()
3. Fair benchmark: Both models trained with AdamW (15 epochs, seed=42) and identical augmentations.
4. Comprehensive testing:
   - Clean accuracy & ROC-AUC on 300 test faces.
   - White-box FGSM & PGD adversarial attacks (eps = [0.0, 0.01, 0.02, 0.04, 0.06]).
   - Social media JPEG compression robustness (qualities = [90, 70, 50, 30]).
   - Single-Branch Ablation Study (Pixel-Only, Texture-Only, Region-Only, vs. Full Fusion).
5. Exports:
   - saved_models/baseline.pt and saved_models/granular_net.pt
   - saved_models/metrics_summary.json
   - saved_models/confusion_matrices.png and saved_models/robustness_curve.png
"""

import os
import sys
import json
import random
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# Configure safe matplotlib cache directory
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders
from src.granules_and_model import GranularDeepfakeNet, SingleLevelBaselineNet


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# 1. ADVERSARIAL ATTACK ENGINES (WHITE-BOX FGSM & PGD IN [0.0, 1.0])
# -----------------------------------------------------------------------------

def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.04,
) -> torch.Tensor:
    """White-box Fast Gradient Sign Method in [0.0, 1.0] pixel units."""
    if epsilon <= 0.0:
        return images.clone().detach()

    images_adv = images.clone().detach().requires_grad_(True)
    out = model(images_adv)
    logits = out[0] if isinstance(out, tuple) else out

    loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    model.zero_grad()
    loss.backward()

    grad_sign = images_adv.grad.data.sign()
    perturbed = images_adv.detach() + epsilon * grad_sign
    return torch.clamp(perturbed, 0.0, 1.0)


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.04,
    alpha: float = 0.01,
    steps: int = 5,
) -> torch.Tensor:
    """White-box Projected Gradient Descent iterative attack in [0.0, 1.0]."""
    if epsilon <= 0.0:
        return images.clone().detach()

    noise = torch.empty_like(images).uniform_(-epsilon, epsilon)
    perturbed = torch.clamp(images + noise, 0.0, 1.0).detach()

    for _ in range(steps):
        perturbed.requires_grad_(True)
        out = model(perturbed)
        logits = out[0] if isinstance(out, tuple) else out

        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        model.zero_grad()
        loss.backward()

        grad_sign = perturbed.grad.data.sign()
        perturbed = perturbed.detach() + alpha * grad_sign

        eta = torch.clamp(perturbed - images, min=-epsilon, max=epsilon)
        perturbed = torch.clamp(images + eta, 0.0, 1.0).detach()

    return perturbed


# -----------------------------------------------------------------------------
# 2. JPEG COMPRESSION ENGINE
# -----------------------------------------------------------------------------

def apply_jpeg_compression_batch(images: torch.Tensor, quality: int = 50) -> torch.Tensor:
    """Applies realistic JPEG compression artifacts to simulate social media encoding."""
    b = images.shape[0]
    compressed_list = []

    for i in range(b):
        img_np = (images[i].cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode(".jpg", img_np, encode_param)
        decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        dec_rgb = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
        tensor_comp = torch.from_numpy(dec_rgb.transpose(2, 0, 1)).float() / 255.0
        compressed_list.append(tensor_comp)

    return torch.stack(compressed_list, dim=0).to(images.device)


# -----------------------------------------------------------------------------
# 3. FAIR TRAINING WITH MULTI-TASK AUXILIARY LOSS & ENTROPY REGULARIZATION
# -----------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 5e-4,
    is_granular: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Trains model for 15 epochs with AdamW, Cosine Annealing, and BCEWithLogitsLoss.
    For Granular model:
      loss = bce_main + 0.25 * (bce_pix + bce_tex + bce_reg) + 0.01 * (w * log(w)).sum(dim=1).mean()
    """
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights = None

    name = "GranularDeepfakeNet" if is_granular else "SingleLevelBaselineNet"
    print(f"\n---> Training {name} ({epochs} epochs, lr={lr}, device={device})...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Identical training data augmentations (25% chance of mild JPEG compression)
            if random.random() < 0.25:
                q = random.randint(60, 90)
                images = apply_jpeg_compression_batch(images, quality=q)

            optimizer.zero_grad()

            if is_granular:
                logits, trust_weights, (aux_pix, aux_tex, aux_reg) = model(images, return_aux=True)
                bce_main = criterion(logits, labels.float())
                bce_pix = criterion(aux_pix, labels.float())
                bce_tex = criterion(aux_tex, labels.float())
                bce_reg = criterion(aux_reg, labels.float())

                aux_loss = 0.25 * (bce_pix + bce_tex + bce_reg)
                # Gate entropy regularization: rewards high entropy (uniform weights)
                gate_entropy = 0.01 * (trust_weights * torch.log(trust_weights + 1e-8)).sum(dim=1).mean()

                loss = bce_main + aux_loss + gate_entropy
            else:
                logits = model(images)
                loss = criterion(logits, labels.float())

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            n_samples += images.size(0)

        scheduler.step()
        epoch_train_loss = total_loss / n_samples
        val_loss, val_acc, _ = evaluate_model(model, val_loader, epsilon=0.0, device=device)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Track best model by validation accuracy
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"  Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.1f}% (Best: {best_val_acc * 100:.1f}%)"
        )

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


# -----------------------------------------------------------------------------
# 4. EVALUATION ENGINE (ACCURACY, ROC-AUC, PRECISION, RECALL, F1)
# -----------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    epsilon: float = 0.0,
    attack_type: str = "fgsm",
    jpeg_quality: Optional[int] = None,
    ablation: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, Dict[str, Any]]:
    """Evaluates model across Clean, FGSM, PGD, JPEG compression, and Ablation settings."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    all_trust = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if jpeg_quality is not None:
            images = apply_jpeg_compression_batch(images, quality=jpeg_quality)
        elif epsilon > 0.0:
            if attack_type.lower() == "fgsm":
                images = fgsm_attack(model, images, labels, epsilon=epsilon)
            elif attack_type.lower() == "pgd":
                images = pgd_attack(model, images, labels, epsilon=epsilon, steps=5)

        with torch.no_grad():
            if isinstance(model, GranularDeepfakeNet):
                out = model(images, ablation=ablation)
            else:
                out = model(images)

            if isinstance(out, tuple):
                logits, trust = out
                all_trust.append(trust.cpu().numpy())
            else:
                logits = out

            loss = criterion(logits, labels.float())
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    n = len(all_targets)
    avg_loss = total_loss / n if n > 0 else 0.0
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])

    mean_trust = None
    if all_trust:
        mean_trust = np.mean(np.concatenate(all_trust, axis=0), axis=0).tolist()

    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "mean_trust_weights": mean_trust,
    }

    return avg_loss, acc, metrics


# -----------------------------------------------------------------------------
# 5. HONEST PRESENTATION CHARTS GENERATION
# -----------------------------------------------------------------------------

def generate_presentation_charts(
    metrics_summary: Dict[str, Any],
    clean_cm: np.ndarray,
    attack_cm: np.ndarray,
    epsilons: List[float],
    fgsm_baseline_acc: List[float],
    fgsm_granular_acc: List[float],
    pgd_baseline_acc: List[float],
    pgd_granular_acc: List[float],
    trust_weight_dynamics: Dict[str, List[float]],
    jpeg_qualities: List[int],
    jpeg_baseline_acc: List[float],
    jpeg_granular_acc: List[float],
    output_dir: str = "saved_models",
):
    """
    Renders presentation figures with honest scaling:
    - Full y-axis [0, 105] on accuracy plot.
    - Full y-axis [0.0, 1.0] on trust weights plot displaying all 3 branches.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    class_names = ["Real (0)", "Fake (1)"]

    # 1. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    disp_clean = ConfusionMatrixDisplay(confusion_matrix=clean_cm, display_labels=class_names)
    disp_clean.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(
        f"Granular Net: Clean Real Faces\nAccuracy: {metrics_summary['granular']['clean']['accuracy'] * 100:.1f}% | AUC: {metrics_summary['granular']['clean']['roc_auc']:.3f}",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    axes[0].set_xlabel("Predicted Label", fontsize=10)
    axes[0].set_ylabel("True Label", fontsize=10)

    disp_attack = ConfusionMatrixDisplay(confusion_matrix=attack_cm, display_labels=class_names)
    disp_attack.plot(ax=axes[1], cmap="Oranges", colorbar=False)
    axes[1].set_title(
        rf"Granular Net: Under FGSM ($\epsilon=0.04$)" + f"\nAccuracy: {metrics_summary['granular']['fgsm_eps_0.04']['accuracy'] * 100:.1f}% | AUC: {metrics_summary['granular']['fgsm_eps_0.04']['roc_auc']:.3f}",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    axes[1].set_xlabel("Predicted Label", fontsize=10)
    axes[1].set_ylabel("True Label", fontsize=10)

    fig.suptitle(
        "Kaggle Dataset: Confusion Matrices (Clean vs. FGSM Attack)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    cm_path = out_path / "confusion_matrices.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"[Chart Saved] Confusion Matrices -> {cm_path}")

    # 2. Comprehensive 2x2 Robustness & Ablation Figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    eps_percent = [e * 100 for e in epsilons]

    # Panel A: Adversarial Robustness Curve (Full y-axis [0, 105])
    ax1.plot(eps_percent, [a * 100 for a in fgsm_granular_acc], "g-o", linewidth=2.4, label="Proposed Granular (FGSM)")
    ax1.plot(eps_percent, [a * 100 for a in fgsm_baseline_acc], "r--s", linewidth=2.0, label="Baseline MobileNetV3 (FGSM)")
    ax1.plot(eps_percent, [a * 100 for a in pgd_granular_acc], "b-^", linewidth=2.2, label="Proposed Granular (PGD)")
    ax1.plot(eps_percent, [a * 100 for a in pgd_baseline_acc], "m--d", linewidth=2.0, label="Baseline MobileNetV3 (PGD)")
    ax1.set_title("A) Adversarial Robustness (Accuracy vs. Perturbation)", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"Perturbation Budget $\epsilon$ (%)", fontsize=10)
    ax1.set_ylabel("Detection Accuracy (%)", fontsize=10)
    ax1.set_ylim([0, 105])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=9)

    # Panel B: Dynamic Granular Trust Weights (Full y-axis [0.0, 1.0], all 3 branches)
    ax2.plot(eps_percent, trust_weight_dynamics["pixel"], "r-o", linewidth=2.2, label=r"Pixel Granule ($w_{pixel}$)")
    ax2.plot(eps_percent, trust_weight_dynamics["texture"], "b-s", linewidth=2.2, label=r"Texture Granule ($w_{texture}$)")
    ax2.plot(eps_percent, trust_weight_dynamics["region"], "g-^", linewidth=2.2, label=r"Facial Region Granule ($w_{region}$)")
    ax2.set_title("B) Adaptive Granular Trust Dynamics under Attack", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Adversarial Noise $\epsilon$ (%)", fontsize=10)
    ax2.set_ylabel("Granular Trust Weight", fontsize=10)
    ax2.set_ylim([0.0, 1.0])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9.5)

    # Panel C: JPEG Compression Robustness
    ax3.plot(jpeg_qualities, [a * 100 for a in jpeg_granular_acc], "g-o", linewidth=2.4, label="Proposed Granular Net")
    ax3.plot(jpeg_qualities, [a * 100 for a in jpeg_baseline_acc], "r--s", linewidth=2.0, label="Baseline MobileNetV3")
    ax3.set_title("C) Social Media Compression Robustness (JPEG Quality)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("JPEG Compression Quality Factor", fontsize=10)
    ax3.set_ylabel("Detection Accuracy (%)", fontsize=10)
    ax3.set_ylim([0, 105])
    ax3.invert_xaxis()
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower left", fontsize=9.5)

    # Panel D: Single-Branch Ablation Study
    ablation_labels = ["Pixel-Only", "Texture-Only", "Region-Only", "Full Granular\nFusion"]
    abl_accs = [
        metrics_summary["ablation"]["pixel_only"]["accuracy"] * 100,
        metrics_summary["ablation"]["texture_only"]["accuracy"] * 100,
        metrics_summary["ablation"]["region_only"]["accuracy"] * 100,
        metrics_summary["ablation"]["full_fusion"]["accuracy"] * 100,
    ]
    colors = ["#EF4444", "#3B82F6", "#10B981", "#8B5CF6"]
    bars = ax4.bar(ablation_labels, abl_accs, color=colors, width=0.55)
    ax4.set_title("D) Single-Branch Ablation Comparison", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Clean Test Accuracy (%)", fontsize=10)
    ax4.set_ylim([0, 105])
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        h = b.get_height()
        ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Robustness & Ablation Analysis: Proposed Granular Fusion vs. Baseline MobileNetV3",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    curve_path = out_path / "robustness_curve.png"
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()
    print(f"[Chart Saved] Robustness Curves & Ablation -> {curve_path}")


# -----------------------------------------------------------------------------
# 6. MAIN PIPELINE EXECUTION
# -----------------------------------------------------------------------------

def run_pipeline():
    print("=" * 70)
    print("  Upgraded Granular Deepfake Detector: Training & Evaluation Pipeline")
    print("  Resolution: 160x160 | Dataset: 2,000 Kaggle Real & Deepfake Faces")
    print("=" * 70)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Execution Engine] Device: {device}")

    # 1. DataLoaders (160x160 resolution)
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="data",
        batch_size=32,
        target_size=160,
        split_ratios=(0.70, 0.15, 0.15),
        random_state=42,
    )

    # 2. Models
    model_baseline = SingleLevelBaselineNet()
    model_granular = GranularDeepfakeNet(emb_dim=64)

    params_b = sum(p.numel() for p in model_baseline.parameters() if p.requires_grad)
    params_g = sum(p.numel() for p in model_granular.parameters() if p.requires_grad)
    print(f"[Model Capacity] Baseline MobileNetV3: {params_b:,} params | Granular Net: {params_g:,} params")

    # 3. Fair Training (15 epochs)
    model_baseline, _ = train_model(
        model=model_baseline,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        lr=5e-4,
        is_granular=False,
        device=device,
    )

    model_granular, _ = train_model(
        model=model_granular,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        lr=5e-4,
        is_granular=True,
        device=device,
    )

    # Save checkpoints
    save_dir = Path("saved_models")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model_baseline.state_dict(), save_dir / "baseline.pt")
    torch.save(model_granular.state_dict(), save_dir / "granular_net.pt")
    print(f"\n[Checkpoints Saved] Model weights saved into {save_dir}/")

    # 4. Adversarial Epsilon Sweep
    epsilons = [0.0, 0.01, 0.02, 0.04, 0.06]
    print("\n---> Running White-Box Adversarial Robustness Sweep:", epsilons)

    fgsm_baseline_acc = []
    fgsm_granular_acc = []
    pgd_baseline_acc = []
    pgd_granular_acc = []
    trust_dynamics = {"pixel": [], "texture": [], "region": []}

    metrics_summary: Dict[str, Any] = {
        "dataset": {"real_count": 1000, "fake_count": 1000, "total": 2000, "test_count": len(test_loader.dataset)},
        "baseline": {"parameters": params_b, "clean": {}, "fgsm_eps_0.04": {}, "pgd_eps_0.04": {}},
        "granular": {"parameters": params_g, "clean": {}, "fgsm_eps_0.04": {}, "pgd_eps_0.04": {}},
        "epsilon_sweep": {
            "epsilons": epsilons,
            "baseline_fgsm": [],
            "granular_fgsm": [],
            "baseline_pgd": [],
            "granular_pgd": [],
        },
        "trust_dynamics": {},
        "jpeg_compression": {},
        "ablation": {},
    }

    clean_cm = None
    attack_cm = None

    for eps in epsilons:
        _, acc_b_fgsm, met_b_fgsm = evaluate_model(model_baseline, test_loader, epsilon=eps, attack_type="fgsm", device=device)
        _, acc_g_fgsm, met_g_fgsm = evaluate_model(model_granular, test_loader, epsilon=eps, attack_type="fgsm", device=device)

        _, acc_b_pgd, met_b_pgd = evaluate_model(model_baseline, test_loader, epsilon=eps, attack_type="pgd", device=device)
        _, acc_g_pgd, met_g_pgd = evaluate_model(model_granular, test_loader, epsilon=eps, attack_type="pgd", device=device)

        fgsm_baseline_acc.append(acc_b_fgsm)
        fgsm_granular_acc.append(acc_g_fgsm)
        pgd_baseline_acc.append(acc_b_pgd)
        pgd_granular_acc.append(acc_g_pgd)

        tw = met_g_fgsm["mean_trust_weights"]
        trust_dynamics["pixel"].append(tw[0])
        trust_dynamics["texture"].append(tw[1])
        trust_dynamics["region"].append(tw[2])

        if eps == 0.0:
            clean_cm = np.array(met_g_fgsm["confusion_matrix"])
            metrics_summary["baseline"]["clean"] = met_b_fgsm
            metrics_summary["granular"]["clean"] = met_g_fgsm

        if abs(eps - 0.04) < 1e-4:
            attack_cm = np.array(met_g_fgsm["confusion_matrix"])
            metrics_summary["baseline"]["fgsm_eps_0.04"] = met_b_fgsm
            metrics_summary["granular"]["fgsm_eps_0.04"] = met_g_fgsm
            metrics_summary["baseline"]["pgd_eps_0.04"] = met_b_pgd
            metrics_summary["granular"]["pgd_eps_0.04"] = met_g_pgd

        print(
            f"  eps={eps:<4} | "
            f"Baseline FGSM Acc: {acc_b_fgsm * 100:5.1f}% -> Granular Acc: {acc_g_fgsm * 100:5.1f}% | "
            f"Trust: [P={tw[0]:.2f}, T={tw[1]:.2f}, R={tw[2]:.2f}]"
        )

    metrics_summary["epsilon_sweep"]["baseline_fgsm"] = fgsm_baseline_acc
    metrics_summary["epsilon_sweep"]["granular_fgsm"] = fgsm_granular_acc
    metrics_summary["epsilon_sweep"]["baseline_pgd"] = pgd_baseline_acc
    metrics_summary["epsilon_sweep"]["granular_pgd"] = pgd_granular_acc
    metrics_summary["trust_dynamics"] = trust_dynamics

    # 5. JPEG Compression Evaluation Sweep
    jpeg_qualities = [90, 70, 50, 30]
    print("\n---> Running JPEG Compression Robustness Sweep:", jpeg_qualities)
    jpeg_baseline_acc = []
    jpeg_granular_acc = []

    for q in jpeg_qualities:
        _, acc_b_q, _ = evaluate_model(model_baseline, test_loader, jpeg_quality=q, device=device)
        _, acc_g_q, _ = evaluate_model(model_granular, test_loader, jpeg_quality=q, device=device)
        jpeg_baseline_acc.append(acc_b_q)
        jpeg_granular_acc.append(acc_g_q)
        print(f"  Quality={q:02d} | Baseline Acc: {acc_b_q * 100:5.1f}% | Granular Acc: {acc_g_q * 100:5.1f}%")

    metrics_summary["jpeg_compression"] = {
        "qualities": jpeg_qualities,
        "baseline_accuracy": jpeg_baseline_acc,
        "granular_accuracy": jpeg_granular_acc,
    }

    # 6. Single-Branch Ablation Study
    print("\n---> Running Single-Branch Ablation Study on Clean Test Set...")
    _, acc_pix, met_pix = evaluate_model(model_granular, test_loader, ablation="pixel", device=device)
    _, acc_tex, met_tex = evaluate_model(model_granular, test_loader, ablation="texture", device=device)
    _, acc_reg, met_reg = evaluate_model(model_granular, test_loader, ablation="region", device=device)
    _, acc_full, met_full = evaluate_model(model_granular, test_loader, ablation=None, device=device)

    print(f"  Pixel-Only Branch Acc : {acc_pix * 100:.1f}% | AUC: {met_pix['roc_auc']:.3f}")
    print(f"  Texture-Only Branch Acc: {acc_tex * 100:.1f}% | AUC: {met_tex['roc_auc']:.3f}")
    print(f"  Region-Only Branch Acc : {acc_reg * 100:.1f}% | AUC: {met_reg['roc_auc']:.3f}")
    print(f"  Full Granular Fusion   : {acc_full * 100:.1f}% | AUC: {met_full['roc_auc']:.3f}")

    metrics_summary["ablation"] = {
        "pixel_only": met_pix,
        "texture_only": met_tex,
        "region_only": met_reg,
        "full_fusion": met_full,
    }

    # Save Metrics JSON
    json_path = save_dir / "metrics_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n[Metrics Saved] Full metrics written to {json_path}")

    # Generate Publication Charts
    generate_presentation_charts(
        metrics_summary=metrics_summary,
        clean_cm=clean_cm,
        attack_cm=attack_cm,
        epsilons=epsilons,
        fgsm_baseline_acc=fgsm_baseline_acc,
        fgsm_granular_acc=fgsm_granular_acc,
        pgd_baseline_acc=pgd_baseline_acc,
        pgd_granular_acc=pgd_granular_acc,
        trust_weight_dynamics=trust_dynamics,
        jpeg_qualities=jpeg_qualities,
        jpeg_baseline_acc=jpeg_baseline_acc,
        jpeg_granular_acc=jpeg_granular_acc,
        output_dir="saved_models",
    )

    print("\n" + "=" * 70)
    print("  TRAINING & EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  Final Granular Clean Test Accuracy : {met_full['accuracy'] * 100:.1f}%")
    print(f"  Final Granular Clean Test ROC-AUC  : {met_full['roc_auc']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()

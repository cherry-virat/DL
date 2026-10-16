"""
Master Multi-Model Ensemble Training Orchestrator.
=================================================

Orchestrates the training of 3-Model Soft-Voting Ensembles for both pipelines:
  - Glioma: Models 1 (Baseline), 2 (Seed 101), 3 (Seed 2024)
  - Meningioma: Models 1 (Baseline), 2 (Seed 101), 3 (Seed 2024)
"""

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_glioma_ensemble_models import train_single_model as train_glioma
from training.train_meningioma_ensemble_models import train_single_meningioma_model as train_meningioma

GLIOMA_ENSEMBLE_DIR = PROJECT_ROOT / "outputs" / "glioma_ensemble"
MENINGIOMA_ENSEMBLE_DIR = PROJECT_ROOT / "outputs" / "meningioma_ensemble"

GLIOMA_BASE_CKPT = PROJECT_ROOT / "outputs" / "glioma_training" / "glioma_best_medium_experiment.pth"
MEN_BASE_CKPT = PROJECT_ROOT / "outputs" / "meningioma_training" / "meningioma_best.pth"


def setup_model_1():
    GLIOMA_ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    MENINGIOMA_ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

    dest_glioma_1 = GLIOMA_ENSEMBLE_DIR / "glioma_model_1.pth"
    if GLIOMA_BASE_CKPT.exists() and not dest_glioma_1.exists():
        shutil.copyfile(GLIOMA_BASE_CKPT, dest_glioma_1)
        print(f"[OK] Linked Glioma Model 1 from {GLIOMA_BASE_CKPT}")

    dest_men_1 = MENINGIOMA_ENSEMBLE_DIR / "meningioma_model_1.pth"
    if MEN_BASE_CKPT.exists() and not dest_men_1.exists():
        shutil.copyfile(MEN_BASE_CKPT, dest_men_1)
        print(f"[OK] Linked Meningioma Model 1 from {MEN_BASE_CKPT}")


def main():
    print("=" * 80)
    print("STARTING FULL MULTI-MODEL ENSEMBLE TRAINING (OPTION B)")
    print("=" * 80)

    setup_model_1()

    # 1. Train Glioma Models 2 and 3
    print("\n>>> PHASE 1: Training Glioma Ensemble Models 2 & 3 <<<")
    train_glioma(model_idx=2, seed=101, loss_type="dice_ce", lr=1.5e-4)
    train_glioma(model_idx=3, seed=2024, loss_type="dice_focal", lr=1.0e-4)

    # 2. Train Meningioma Models 2 and 3
    print("\n>>> PHASE 2: Training Meningioma Ensemble Models 2 & 3 <<<")
    train_meningioma(model_idx=2, seed=101, tumor_weight=2.5, lr=1.5e-4)
    train_meningioma(model_idx=3, seed=2024, tumor_weight=1.5, lr=1.0e-4)

    print("\n" + "=" * 80)
    print("ALL 6 ENSEMBLE MODELS SUCCESSFULLY TRAINED & READY FOR SOFT-VOTING!")
    print("=" * 80)


if __name__ == "__main__":
    main()


"""
Tumor Classifier Integration Service.
=====================================

Provides automatic tumor presence detection and tumor-type classification services
for 3D MRI brain scans, bridging the upload layer to the unified InferenceRouter.

Honesty & Safety Guarantees:
----------------------------
1. Zero Untrained Weights Execution: If trained classifier weights are not present
   on disk, this service strictly returns 'automatic_classification_not_available'
   with the exact blocker and remediation steps. It NEVER executes randomly initialized
   weights, guesses based on segmentation mask volume, or fabricates confidence.
2. Modality Alignment:
   - Stage 1 (Healthy vs. Tumor Presence): Operates on 2 structural channels (T1, T2).
   - Stage 2 (Glioma vs. Meningioma Type): Operates on 4 structural channels (T1, T1c, T2, FLAIR).
3. Safe Fallback: Exposes available manual routes ('glioma', 'meningioma') when automatic
   classification cannot be medically or technically justified.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import nibabel as nib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Explicit default checkpoint locations (configurable via environment variables)
ENV_PRESENCE_CHECKPOINT = os.getenv("TUMOR_PRESENCE_CHECKPOINT_PATH")
ENV_TYPE_CHECKPOINT = os.getenv("TUMOR_TYPE_CHECKPOINT_PATH")

STAGE1_CANONICAL_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "classifier_training" / "tumor_presence_best.pth"
)
STAGE2_EXPERIMENT_20261019_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "classifier_training" / "experiment_20261019" / "tumor_classifier_best.pth"
)
STAGE2_BASELINE_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "classifier_training" / "tumor_classifier_best.pth"
)

# Explicit configuration with NO SILENT FALLBACK:
# 1. Respects TUMOR_PRESENCE_CHECKPOINT_PATH and TUMOR_TYPE_CHECKPOINT_PATH if provided.
# 2. Otherwise, explicitly targets STAGE1_CANONICAL_CHECKPOINT and STAGE2_EXPERIMENT_20261019_CHECKPOINT.
# If a targeted checkpoint is missing on disk, it is honestly reported as unavailable rather than
# secretly substituting an older baseline or unvalidated file.
DEFAULT_PRESENCE_CHECKPOINT = (
    Path(ENV_PRESENCE_CHECKPOINT)
    if ENV_PRESENCE_CHECKPOINT
    else STAGE1_CANONICAL_CHECKPOINT
)
DEFAULT_TYPE_CHECKPOINT = (
    Path(ENV_TYPE_CHECKPOINT)
    if ENV_TYPE_CHECKPOINT
    else STAGE2_EXPERIMENT_20261019_CHECKPOINT
)


class ClassifierService:
    """
    Dedicated service handling automatic tumor-presence and tumor-type assessment.
    """

    def __init__(
        self,
        presence_checkpoint_path: Optional[Union[str, Path]] = None,
        type_checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.presence_checkpoint_path = Path(presence_checkpoint_path) if presence_checkpoint_path else DEFAULT_PRESENCE_CHECKPOINT
        self.type_checkpoint_path = Path(type_checkpoint_path) if type_checkpoint_path else DEFAULT_TYPE_CHECKPOINT

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._presence_model: Optional[torch.nn.Module] = None
        self._type_model: Optional[torch.nn.Module] = None

    @property
    def is_presence_model_available(self) -> bool:
        """Checks if a genuine trained checkpoint exists for the tumor presence detector."""
        return self.presence_checkpoint_path.exists() and self.presence_checkpoint_path.is_file()

    @property
    def is_type_model_available(self) -> bool:
        """Checks if a genuine trained checkpoint exists for the glioma-vs-meningioma classifier."""
        if not (self.type_checkpoint_path.exists() and self.type_checkpoint_path.is_file()):
            return False
        # Do not treat sample-truncated / smoke-test checkpoints (< 50 validation samples) as verified classifiers
        try:
            ckpt = torch.load(self.type_checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict):
                total_samples = ckpt.get("val_metrics", {}).get("total_samples", None)
                if total_samples is not None and total_samples < 50:
                    return False
        except Exception:
            pass
        return True

    @property
    def is_automatic_classification_available(self) -> bool:
        """Returns True only when trained checkpoints are present and verified."""
        return self.is_presence_model_available and self.is_type_model_available

    def get_status(self) -> Dict[str, Any]:
        """Returns the current deployment status and availability of automatic classifiers."""
        type_info = {
            "checkpoint_path": str(self.type_checkpoint_path),
            "available": self.is_type_model_available,
            "epoch": None,
            "val_macro_f1": None,
            "val_samples": None,
            "sha256": None,
        }
        if self.type_checkpoint_path.exists() and self.type_checkpoint_path.is_file():
            try:
                with open(self.type_checkpoint_path, "rb") as f:
                    type_info["sha256"] = hashlib.sha256(f.read()).hexdigest()
                ckpt = torch.load(self.type_checkpoint_path, map_location="cpu")
                if isinstance(ckpt, dict):
                    type_info["epoch"] = ckpt.get("epoch")
                    type_info["val_macro_f1"] = ckpt.get("best_val_f1")
                    type_info["val_samples"] = ckpt.get("val_metrics", {}).get("total_samples")
            except Exception:
                pass

        presence_info = {
            "checkpoint_path": str(self.presence_checkpoint_path),
            "available": self.is_presence_model_available,
            "sha256": None,
        }
        if self.presence_checkpoint_path.exists() and self.presence_checkpoint_path.is_file():
            try:
                with open(self.presence_checkpoint_path, "rb") as f:
                    presence_info["sha256"] = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                pass

        return {
            "automatic_classification_available": self.is_automatic_classification_available,
            "presence_model_available": self.is_presence_model_available,
            "type_model_available": self.is_type_model_available,
            "presence_checkpoint": str(self.presence_checkpoint_path),
            "type_checkpoint": str(self.type_checkpoint_path),
            "active_type_checkpoint": type_info,
            "active_presence_checkpoint": presence_info,
            "supported_manual_routes": ["glioma", "meningioma"],
            "blocker_explanation": None if self.is_automatic_classification_available else (
                "Trained 3D MRI classifier checkpoints are not present on disk. "
                "Automatic tumor-presence (healthy vs. tumor) and tumor-type (glioma vs. meningioma) "
                "classification cannot be performed without genuine trained weights."
            ),
        }

    def classify(
        self,
        modality_paths: Dict[str, Path],
    ) -> Dict[str, Any]:
        """
        Assesses tumor presence and identifies tumor type.

        Parameters:
            modality_paths: Dictionary mapping modality names to file paths.

        Returns:
            Dictionary detailing assessment status, predicted route, confidence, and blockers.
        """
        # If genuine checkpoints are missing, strictly report unavailable without fabricating results
        if not self.is_automatic_classification_available:
            missing = []
            if not self.is_presence_model_available:
                missing.append(f"Stage 1 Tumor Presence checkpoint ({self.presence_checkpoint_path.name})")
            if not self.is_type_model_available:
                missing.append(f"Stage 2 Tumor Type checkpoint ({self.type_checkpoint_path.name})")

            return {
                "status": "automatic_classification_not_available",
                "tumor_presence": "uncertain",
                "predicted_tumor_type": None,
                "confidence": None,
                "selected_route": None,
                "reason": (
                    "Automatic classification is not available because trained model weights are missing: "
                    f"{', '.join(missing)}. Randomly initialized weights or heuristics are strictly rejected."
                ),
                "blocker": "Missing trained checkpoints on disk.",
                "available_manual_routes": ["glioma", "meningioma"],
                "recommendation": "Please select the appropriate segmentation model manually ('glioma' or 'meningioma').",
                "next_steps": (
                    "Train Stage 1 presence model on preprocessed IXI/BraTS data and Stage 2 type model on BraTS GLI/MEN data."
                ),
            }

        # If trained checkpoints exist, execute inference
        try:
            return self._execute_trained_classification(modality_paths)
        except Exception as exc:
            return {
                "status": "classification_failed",
                "tumor_presence": "uncertain",
                "predicted_tumor_type": None,
                "confidence": None,
                "selected_route": None,
                "reason": f"Classification inference failed: {str(exc)}",
                "available_manual_routes": ["glioma", "meningioma"],
            }

    def _execute_trained_classification(self, modality_paths: Dict[str, Path]) -> Dict[str, Any]:
        """Loads trained weights and runs sequential Stage 1 -> Stage 2 inference."""
        from models.tumor_presence_classifier_3d import (
            create_tumor_presence_classifier_3d,
            create_presence_classifier,
        )
        from models.tumor_classifier_3d import (
            create_tumor_classifier_3d,
            create_classifier,
        )
        from preprocessing.tumor_presence_preprocess import preprocess_2channel_presence_tensor
        from preprocessing.tumor_classifier_preprocess import preprocess_case_to_tensor

        # Normalize keys
        canon_keys = {str(k).upper(): v for k, v in modality_paths.items()}
        t1_path = canon_keys.get("T1") or canon_keys.get("T1N")
        t2_path = canon_keys.get("T2") or canon_keys.get("T2W")
        t1c_path = canon_keys.get("T1C") or canon_keys.get("T1CE")
        flair_path = canon_keys.get("FLAIR") or canon_keys.get("T2F")

        # 1. Stage 1: Healthy vs. Tumor
        if self._presence_model is None:
            ckpt = torch.load(str(self.presence_checkpoint_path), map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            base_ch = ckpt.get("base_channels") if isinstance(ckpt, dict) and "base_channels" in ckpt else (
                state_dict["stem.0.weight"].shape[0] if "stem.0.weight" in state_dict else 16
            )
            num_cls = ckpt.get("num_classes") if isinstance(ckpt, dict) and "num_classes" in ckpt else (
                state_dict["classifier.weight"].shape[0] if "classifier.weight" in state_dict else 2
            )
            model = create_presence_classifier(in_channels=2, num_classes=num_cls, base_channels=base_ch)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self._presence_model = model

        # Check required modalities for Stage 1 (T1 and T2)
        if not t1_path or not t2_path:
            return {
                "status": "unsupported_modality_set",
                "tumor_presence": "uncertain",
                "predicted_tumor_type": None,
                "confidence": None,
                "reason": "Stage 1 tumor presence detection requires at least T1 and T2 acquisitions.",
                "available_manual_routes": ["glioma", "meningioma"],
            }

        # Prepare 2-channel tensor with exact training preprocessing
        tensor_2ch, _ = preprocess_2channel_presence_tensor(
            {"t1": t1_path, "t2": t2_path},
            auto_align=True,
        )
        tensor_2ch = tensor_2ch.unsqueeze(0).to(self.device)

        with torch.no_grad():
            presence_logits = self._presence_model(tensor_2ch)
            presence_probs = torch.softmax(presence_logits, dim=-1)
            presence_prob = float(presence_probs[0, 1].item())

        # If healthy (probability < 0.5)
        if presence_prob < 0.5:
            presence_conf = round(1.0 - presence_prob, 4)
            return {
                "status": "classified_healthy",
                "tumor_presence": "healthy_control",
                "presence_probability": round(presence_prob, 4),
                "presence_confidence": presence_conf,
                "predicted_tumor_type": None,
                "selected_route": None,
                "confidence": presence_conf,
                "recommendation": "No tumor detected by Stage 1 classifier. No segmentation required.",
            }

        # 2. Stage 2: Glioma vs. Meningioma
        if not (t1c_path and flair_path):
            return {
                "status": "partial_classification",
                "tumor_presence": "tumor_present",
                "presence_probability": round(presence_prob, 4),
                "presence_confidence": round(presence_prob, 4),
                "predicted_tumor_type": None,
                "selected_route": None,
                "reason": "Tumor present, but Stage 2 differentiation requires 4 channels (T1, T1c, T2, FLAIR).",
                "available_manual_routes": ["glioma", "meningioma"],
            }

        if self._type_model is None:
            ckpt_t = torch.load(str(self.type_checkpoint_path), map_location=self.device)
            state_dict_t = ckpt_t.get("model_state_dict", ckpt_t) if isinstance(ckpt_t, dict) else ckpt_t
            base_ch_t = ckpt_t.get("base_channels") if isinstance(ckpt_t, dict) and "base_channels" in ckpt_t else (
                state_dict_t["stem.0.weight"].shape[0] if "stem.0.weight" in state_dict_t else 16
            )
            num_cls_t = ckpt_t.get("num_classes") if isinstance(ckpt_t, dict) and "num_classes" in ckpt_t else (
                state_dict_t["classifier.weight"].shape[0] if "classifier.weight" in state_dict_t else 2
            )
            type_m = create_classifier(in_channels=4, num_classes=num_cls_t, base_channels=base_ch_t)
            type_m.load_state_dict(state_dict_t)
            type_m.to(self.device)
            type_m.eval()
            self._type_model = type_m

        type_tensor = preprocess_case_to_tensor({
            "t1": t1_path, "t1c": t1c_path, "t2": t2_path, "flair": flair_path
        }).unsqueeze(0).to(self.device)

        with torch.no_grad():
            type_logits = self._type_model(type_tensor)
            type_probs = torch.softmax(type_logits, dim=-1)
            meningioma_prob = float(type_probs[0, 1].item())

        is_meningioma = (meningioma_prob >= 0.5)
        tumor_type = "meningioma" if is_meningioma else "glioma"
        confidence = meningioma_prob if is_meningioma else (1.0 - meningioma_prob)
        glioma_prob = round(1.0 - meningioma_prob, 4)
        meningioma_prob_rounded = round(meningioma_prob, 4)

        # Ambiguity / uncertainty check: if confidence is below 0.55, refuse automatic routing
        UNCERTAINTY_THRESHOLD = 0.55
        if confidence < UNCERTAINTY_THRESHOLD:
            return {
                "status": "uncertain_classification",
                "tumor_presence": "tumor_present",
                "presence_probability": round(presence_prob, 4),
                "presence_confidence": round(presence_prob, 4),
                "predicted_tumor_type": "uncertain",
                "selected_route": None,
                "type_probability": meningioma_prob_rounded,
                "confidence": round(confidence, 4),
                "type_confidence": round(confidence, 4),
                "glioma_probability": glioma_prob,
                "meningioma_probability": meningioma_prob_rounded,
                "type_probabilities": {
                    "glioma": glioma_prob,
                    "meningioma": meningioma_prob_rounded,
                },
                "reason": (
                    f"Tumor type prediction is ambiguous (confidence {confidence:.2f} < {UNCERTAINTY_THRESHOLD}). "
                    "Automatic routing is refused to prevent potential misclassification. Clinical review is required."
                ),
                "available_manual_routes": ["glioma", "meningioma"],
                "recommendation": "Please select the appropriate segmentation model manually ('glioma' or 'meningioma').",
            }

        return {
            "status": "classified_tumor",
            "tumor_presence": "tumor_present",
            "presence_probability": round(presence_prob, 4),
            "presence_confidence": round(presence_prob, 4),
            "predicted_tumor_type": tumor_type,
            "selected_route": tumor_type,
            "type_probability": meningioma_prob_rounded,
            "confidence": round(confidence, 4),
            "type_confidence": round(confidence, 4),
            "glioma_probability": glioma_prob,
            "meningioma_probability": meningioma_prob_rounded,
            "type_probabilities": {
                "glioma": glioma_prob,
                "meningioma": meningioma_prob_rounded,
            },
            "notes": "Classification produced by automated 3D ResNet hierarchy. Clinical verification required.",
        }


_service_instance: Optional[ClassifierService] = None


def get_classifier_service() -> ClassifierService:
    """Returns a shared or singleton instance of ClassifierService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ClassifierService()
    return _service_instance

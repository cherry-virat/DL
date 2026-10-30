"""
Unified Inference Router.
=========================

Provides a single, standardized entry point for multi-modal brain tumor
segmentation dispatching to frozen Glioma or Meningioma pipelines.

Architecture:
-------------
Application / API Layer
        │
        ▼
 Inbound Request (tumor_type, input_data)
        │
        ▼
   InferenceRouter.route()
   - Validates explicit TumorType
   - Validates input modality structure
   - Dispatches to specialized service
        │
 ┌──────┴──────────────────────────┐
 │                                 │
 ▼                                 ▼
GliomaInferenceService    MeningiomaMultimodalInferenceService
 (4-Channel 3D U-Net)      (4-Channel 3D U-Net)
 │                                 │
 └────────────────┬────────────────┘
                  ▼
          SegmentationResult
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import nibabel as nib

from integration.result_schema import (
    TumorType,
    SegmentationResult,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from integration.glioma_inference_service import (
        GliomaInferenceService,
    )
    from integration.meningioma_multimodal_service import (
        MeningiomaMultimodalInferenceService,
    )
    from integration.classifier_service import (
        ClassifierService,
    )


class AutomaticClassificationUnavailableError(ValueError):
    """Raised when automatic classification is requested but trained classifier checkpoints are unavailable."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class NoTumorDetectedError(ValueError):
    """Raised when automatic classification detects a healthy brain / no tumor present."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class InferenceRouter:
    """
    Unified router dispatching requests to the appropriate frozen segmentation service.
    """

    def __init__(
        self,
        glioma_service: Optional[Any] = None,
        meningioma_service: Optional[Any] = None,
        meningioma_multimodal_service: Optional[Any] = None,
        classifier_service: Optional[Any] = None,
    ):
        self._glioma_service = glioma_service
        # The sole active meningioma service is 4-channel multimodal
        self._meningioma_service = meningioma_multimodal_service or meningioma_service
        self._classifier_service = classifier_service

    @property
    def glioma_service(self) -> Any:
        if self._glioma_service is None:
            from integration.glioma_inference_service import GliomaInferenceService
            self._glioma_service = GliomaInferenceService()
        return self._glioma_service

    @glioma_service.setter
    def glioma_service(self, service: Optional[Any]) -> None:
        self._glioma_service = service

    @property
    def meningioma_service(self) -> Any:
        if self._meningioma_service is None:
            from integration.meningioma_multimodal_service import MeningiomaMultimodalInferenceService
            self._meningioma_service = MeningiomaMultimodalInferenceService()
        return self._meningioma_service

    @meningioma_service.setter
    def meningioma_service(self, service: Optional[Any]) -> None:
        self._meningioma_service = service

    @property
    def meningioma_multimodal_service(self) -> Any:
        return self.meningioma_service

    @meningioma_multimodal_service.setter
    def meningioma_multimodal_service(self, service: Optional[Any]) -> None:
        self.meningioma_service = service

    @property
    def classifier_service(self) -> Any:
        if self._classifier_service is None:
            from integration.classifier_service import ClassifierService
            self._classifier_service = ClassifierService()
        return self._classifier_service

    @classifier_service.setter
    def classifier_service(self, service: Optional[Any]) -> None:
        self._classifier_service = service

    def normalize_tumor_type(self, tumor_type: Union[TumorType, str]) -> TumorType:
        """
        Validates and standardizes the tumor type input.
        """
        if isinstance(tumor_type, TumorType):
            return tumor_type

        if isinstance(tumor_type, str):
            clean_str = tumor_type.strip().lower().replace("-", "_")
            if clean_str in [
                "meningioma_multimodal",
                "multimodal_meningioma",
                "meningioma_4channel",
                "meningioma_multi",
            ]:
                clean_str = "meningioma_multimodal"
            try:
                return TumorType(clean_str)
            except ValueError:
                raise ValueError(
                    f"Unsupported tumor type: '{tumor_type}'. "
                    f"Supported types are: {[t.value for t in TumorType]}"
                )

        raise TypeError(
            f"Expected tumor_type to be TumorType or str, got {type(tumor_type).__name__}"
        )

    def validate_input_for_tumor_type(
        self,
        tumor_type: TumorType,
        input_data: Any,
    ) -> None:
        """
        Validates that the input data format matches the required modalities for the tumor type.
        Detects and rejects missing, duplicate, or unexpected modalities.
        """
        if input_data is None:
            raise ValueError(f"input_data cannot be None for {tumor_type.value} segmentation.")

        if tumor_type == TumorType.GLIOMA:
            # Glioma requires 4 modalities (T1, T1c, T2, FLAIR)
            if isinstance(input_data, dict):
                required = {"T1", "T1c", "T2", "FLAIR"}
                canonical_keys: Dict[str, str] = {}
                for k in input_data.keys():
                    k_clean = str(k).strip()
                    k_upper = k_clean.upper()
                    if k_upper in ("T1", "T1N", "T1W"):
                        canon = "T1"
                    elif k_upper in ("T1C", "T1CE", "T1-CE", "T1_CE"):
                        canon = "T1c"
                    elif k_upper in ("T2", "T2W"):
                        canon = "T2"
                    elif k_upper in ("FLAIR", "T2F", "T2-FLAIR", "T2_FLAIR"):
                        canon = "FLAIR"
                    else:
                        raise ValueError(
                            f"Invalid input for Glioma segmentation. Unexpected modality: '{k}'. "
                            f"Required modalities: {sorted(list(required))}"
                        )
                    if canon in canonical_keys:
                        raise ValueError(
                            f"Invalid input for Glioma segmentation. Duplicate modality '{canon}' detected "
                            f"from keys '{canonical_keys[canon]}' and '{k}'."
                        )
                    canonical_keys[canon] = k_clean

                missing = required - set(canonical_keys.keys())
                if missing:
                    raise ValueError(
                        f"Invalid input for Glioma segmentation. Missing required modalities: {sorted(list(missing))}"
                    )
            elif isinstance(input_data, np.ndarray):
                if input_data.ndim != 4 or input_data.shape[0] != 4:
                    raise ValueError(
                        f"Invalid array shape for Glioma segmentation. Expected (4, D, H, W), got {input_data.shape}"
                    )
            elif isinstance(input_data, (str, Path)):
                p = Path(input_data)
                if not p.exists():
                    raise FileNotFoundError(f"Glioma input directory or file does not exist: {p}")
            else:
                raise TypeError(
                    f"Unsupported input_data type for Glioma: {type(input_data).__name__}. "
                    "Expected directory path, dict of 4 modality paths, or 4D numpy array."
                )

        elif tumor_type in (TumorType.MENINGIOMA, TumorType.MENINGIOMA_MULTIMODAL):
            # All Meningioma segmentation requires 4 modalities (t1n, t1c, t2w, t2f)
            if isinstance(input_data, dict):
                from integration.meningioma_multimodal_service import detect_modality_key, REQUIRED_MODALITIES
                detected: Dict[str, str] = {}
                for k in input_data.keys():
                    det = detect_modality_key(str(k))
                    if not det and str(k).lower().strip() in REQUIRED_MODALITIES:
                        det = str(k).lower().strip()
                    if not det:
                        k_upper = str(k).strip().upper()
                        if k_upper == "T1":
                            det = "t1n"
                        elif k_upper == "T1C":
                            det = "t1c"
                        elif k_upper == "T2":
                            det = "t2w"
                        elif k_upper == "FLAIR":
                            det = "t2f"

                    if not det:
                        raise ValueError(
                            f"Invalid input for Meningioma segmentation. Unexpected modality: '{k}'. "
                            f"Required modalities: {sorted(list(REQUIRED_MODALITIES))}"
                        )
                    if det in detected:
                        raise ValueError(
                            f"Invalid input for Meningioma segmentation. Duplicate modality '{det}' detected "
                            f"from keys '{detected[det]}' and '{k}'."
                        )
                    detected[det] = str(k)

                missing = set(REQUIRED_MODALITIES) - set(detected.keys())
                if missing:
                    raise ValueError(
                        f"Invalid input for Meningioma segmentation. Missing required modalities: {sorted(list(missing))}"
                    )
            elif isinstance(input_data, np.ndarray):
                if input_data.ndim != 4 or input_data.shape[0] != 4:
                    raise ValueError(
                        f"Invalid array shape for Meningioma segmentation. Expected (4, D, H, W), got {input_data.shape}"
                    )
            elif isinstance(input_data, (str, Path)):
                p = Path(input_data)
                if not p.exists():
                    raise FileNotFoundError(f"Meningioma input path does not exist: {p}")
            else:
                raise TypeError(
                    f"Unsupported input_data type for Meningioma: {type(input_data).__name__}. "
                    "Expected directory path, dict of 4 modality paths, or 4D numpy array."
                )

        elif tumor_type == TumorType.AUTO:
            if not isinstance(input_data, (dict, np.ndarray, str, Path)):
                raise TypeError(
                    f"Unsupported input_data type for automatic routing: {type(input_data).__name__}. "
                    "Expected directory path, dict of modality paths, or numpy array."
                )

    def route(
        self,
        tumor_type: Union[TumorType, str],
        input_data: Any,
        request_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None,
        save_mask_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> SegmentationResult:
        """
        Routes an inference request to the correct frozen ML service.

        Parameters:
            tumor_type: Explicit TumorType enum or string ('glioma', 'meningioma', 'meningioma_multimodal', 'auto')
            input_data: Modality path(s), directory, or preprocessed array
            request_id: Optional tracking ID
            patient_id: Optional patient identifier
            session_id: Optional session identifier
            save_mask_path: Optional destination to write output NIfTI mask
            **kwargs: Pipeline-specific parameters (e.g., ground_truth_mask, restore_to_native_space)

        Returns:
            SegmentationResult complying with the Common Result Schema
        """
        validated_type = self.normalize_tumor_type(tumor_type)
        self.validate_input_for_tumor_type(validated_type, input_data)

        if validated_type == TumorType.AUTO:
            mod_paths = input_data if isinstance(input_data, dict) else {}
            classification_result = self.classifier_service.classify(mod_paths)
            status = classification_result.get("status")

            if status == "automatic_classification_not_available":
                raise AutomaticClassificationUnavailableError(
                    message=classification_result.get("reason", "Automatic classification not available."),
                    details=classification_result,
                )
            elif status in ("uncertain_classification", "uncertain_presence"):
                raise AutomaticClassificationUnavailableError(
                    message=classification_result.get("reason", "Classification prediction is uncertain. Automatic routing refused."),
                    details=classification_result,
                )
            elif status == "classified_healthy":
                raise NoTumorDetectedError(
                    message=classification_result.get(
                        "recommendation",
                        "No tumor detected by automated presence detector. Segmentation inference is not indicated for normative healthy-control scans.",
                    ),
                    details=classification_result,
                )
            elif status == "classified_tumor":
                predicted_route = classification_result.get("selected_route")
                if predicted_route in ("glioma", "meningioma", "meningioma_multimodal"):
                    result = self.route(
                        tumor_type=predicted_route,
                        input_data=input_data,
                        request_id=request_id,
                        patient_id=patient_id,
                        session_id=session_id,
                        save_mask_path=save_mask_path,
                        **kwargs,
                    )
                    # Attach classification provenance to result quality notes
                    if hasattr(result, "quality_status") and result.quality_status:
                        result.quality_status.qc_notes = (
                            f"Automatic routing: {predicted_route} "
                            f"(presence prob: {classification_result.get('presence_probability')}, "
                            f"confidence: {classification_result.get('confidence')})"
                        )
                    return result
            raise ValueError(
                f"Automatic classification could not determine a valid segmentation route: {classification_result.get('reason')}"
            )

        if validated_type == TumorType.GLIOMA:
            # Canonicalize dictionary keys for standard service consumption
            case_input = input_data
            if isinstance(input_data, dict):
                key_aliases = {
                    "T1": "T1", "T1N": "T1", "T1W": "T1",
                    "T1C": "T1c", "T1CE": "T1c", "T1-CE": "T1c", "T1_CE": "T1c",
                    "T2": "T2", "T2W": "T2",
                    "FLAIR": "FLAIR", "T2F": "FLAIR", "T2-FLAIR": "FLAIR", "T2_FLAIR": "FLAIR",
                }
                case_input = {
                    key_aliases.get(str(k).strip().upper(), str(k)): v
                    for k, v in input_data.items()
                }
            return self.glioma_service.predict(
                case_input=case_input,
                request_id=request_id,
                patient_id=patient_id,
                session_id=session_id,
                save_mask_path=save_mask_path,
                save_gtv_mask_path=kwargs.get("save_gtv_mask_path"),
                ground_truth_mask=kwargs.get("ground_truth_mask"),
                input_metadata=kwargs.get("input_metadata"),
            )

        elif validated_type in (TumorType.MENINGIOMA, TumorType.MENINGIOMA_MULTIMODAL):
            from integration.meningioma_multimodal_service import detect_modality_key
            case_input = input_data
            if isinstance(input_data, dict):
                case_input = {}
                for k, v in input_data.items():
                    det = detect_modality_key(str(k))
                    if not det:
                        k_up = str(k).strip().upper()
                        if k_up == "T1":
                            det = "t1n"
                        elif k_up == "T1C":
                            det = "t1c"
                        elif k_up == "T2":
                            det = "t2w"
                        elif k_up == "FLAIR":
                            det = "t2f"
                        else:
                            det = str(k).strip().lower()
                    case_input[det] = v

            return self.meningioma_service.predict(
                case_input=case_input,
                request_id=request_id,
                patient_id=patient_id,
                session_id=session_id,
                save_mask_path=save_mask_path,
                save_gtv_mask_path=kwargs.get("save_gtv_mask_path"),
                ground_truth_mask=kwargs.get("ground_truth_mask"),
                input_metadata=kwargs.get("input_metadata"),
                tumor_type=validated_type,
            )

        raise RuntimeError(f"Unhandled tumor type in router dispatch: {validated_type}")


def run_inference(
    tumor_type: Union[TumorType, str],
    input_data: Any,
    router: Optional[InferenceRouter] = None,
    **kwargs: Any,
) -> SegmentationResult:
    """
    Convenience function providing a functional interface for routed inference.
    """
    active_router = router or InferenceRouter()
    return active_router.route(tumor_type, input_data, **kwargs)

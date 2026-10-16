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
from integration.glioma_inference_service import (
    GliomaInferenceService,
)
from integration.meningioma_multimodal_service import (
    MeningiomaMultimodalInferenceService,
)


class InferenceRouter:
    """
    Unified router dispatching requests to the appropriate frozen segmentation service.
    """

    def __init__(
        self,
        glioma_service: Optional[GliomaInferenceService] = None,
        meningioma_service: Optional[Any] = None,
        meningioma_multimodal_service: Optional[MeningiomaMultimodalInferenceService] = None,
    ):
        self._glioma_service = glioma_service
        # The sole active meningioma service is 4-channel multimodal
        self._meningioma_service = meningioma_multimodal_service or meningioma_service

    @property
    def glioma_service(self) -> GliomaInferenceService:
        if self._glioma_service is None:
            self._glioma_service = GliomaInferenceService()
        return self._glioma_service

    @property
    def meningioma_service(self) -> MeningiomaMultimodalInferenceService:
        if self._meningioma_service is None:
            self._meningioma_service = MeningiomaMultimodalInferenceService()
        return self._meningioma_service

    @property
    def meningioma_multimodal_service(self) -> MeningiomaMultimodalInferenceService:
        return self.meningioma_service

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
        """
        if input_data is None:
            raise ValueError(f"input_data cannot be None for {tumor_type.value} segmentation.")

        if tumor_type == TumorType.GLIOMA:
            # Glioma requires 4 modalities (T1, T1c, T2, FLAIR)
            if isinstance(input_data, dict):
                required = {"T1", "T1c", "T2", "FLAIR"}
                missing = required - set(input_data.keys())
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
                detected = set()
                for k in input_data.keys():
                    det = detect_modality_key(str(k))
                    if det:
                        detected.add(det)
                    elif str(k).lower().strip() in REQUIRED_MODALITIES:
                        detected.add(str(k).lower().strip())
                missing = set(REQUIRED_MODALITIES) - detected
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
            tumor_type: Explicit TumorType enum or string ('glioma', 'meningioma', 'meningioma_multimodal')
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

        if validated_type == TumorType.GLIOMA:
            # Normalize dictionary if needed
            case_input = input_data
            if isinstance(input_data, dict):
                # Ensure standard case capitalization
                case_input = {k: v for k, v in input_data.items()}
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
            return self.meningioma_service.predict(
                case_input=input_data,
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


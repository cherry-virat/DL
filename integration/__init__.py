"""
Integration Package for Multi-Modal Brain Tumor Segmentation.

This package provides unified schemas, data contracts, thin inference
services, and the unified inference router connecting the frozen Glioma and
Meningioma ML pipelines to downstream clinical APIs and 3D visualization systems.
"""

from integration.result_schema import (
    TumorType,
    PipelineStatus,
    RequestMetadata,
    InputMetadata,
    MaskReference,
    VolumetricSummary,
    GliomaSubregionVolumes,
    GliomaSubregionMasks,
    GliomaMeasurements,
    MeningiomaMeasurements,
    InferenceQualityStatus,
    OfflineEvaluationMetrics,
    SegmentationResult,
)
from integration.glioma_inference_service import (
    GliomaInferenceService,
    run_glioma_inference,
)
from integration.meningioma_inference_service import (
    MeningiomaInferenceService,
    run_meningioma_inference,
)
from integration.meningioma_multimodal_service import (
    MeningiomaMultimodalInferenceService,
    run_meningioma_multimodal_inference,
)
from integration.inference_router import (
    InferenceRouter,
    run_inference,
)

__all__ = [
    "TumorType",
    "PipelineStatus",
    "RequestMetadata",
    "InputMetadata",
    "MaskReference",
    "VolumetricSummary",
    "GliomaSubregionVolumes",
    "GliomaSubregionMasks",
    "GliomaMeasurements",
    "MeningiomaMeasurements",
    "InferenceQualityStatus",
    "OfflineEvaluationMetrics",
    "SegmentationResult",
    "GliomaInferenceService",
    "run_glioma_inference",
    "MeningiomaInferenceService",
    "run_meningioma_inference",
    "MeningiomaMultimodalInferenceService",
    "run_meningioma_multimodal_inference",
    "InferenceRouter",
    "run_inference",
]

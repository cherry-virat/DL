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
from integration.inference_router import (
    InferenceRouter,
    run_inference,
)

# Lazy exports for heavy inference services (PEP 562) to prevent eager PyTorch/MONAI loading
_LAZY_EXPORTS = {
    "GliomaInferenceService": "integration.glioma_inference_service",
    "run_glioma_inference": "integration.glioma_inference_service",
    "MeningiomaInferenceService": "integration.meningioma_inference_service",
    "run_meningioma_inference": "integration.meningioma_inference_service",
    "MeningiomaMultimodalInferenceService": "integration.meningioma_multimodal_service",
    "run_meningioma_multimodal_inference": "integration.meningioma_multimodal_service",
}

def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib
        module = importlib.import_module(_LAZY_EXPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'integration' has no attribute '{name}'")

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

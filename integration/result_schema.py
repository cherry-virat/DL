"""
Common Result Schema for Multi-Modal Brain Tumor Segmentation.
==============================================================

Provides unified, type-safe data contracts representing segmentation outputs,
volumetric quantifications, spatial metadata, and quality status for both
frozen ML pipelines:
  1. Glioma (4-channel multi-class: WT, TC, ET)
  2. Meningioma (1-channel binary: GTV)

Design Principles:
------------------
- Non-destructive to existing frozen ML pipelines.
- Supports native-space and resampled spatial representations.
- Separates clinical inference outputs from optional offline evaluation metrics.
- Enforces data integrity (zero fake validation Dice for Meningioma).
- Direct JSON serializability for RESTful APIs and frontend 3D visualizers.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import datetime
import json


class TumorType(str, Enum):
    """Supported pathology types."""
    GLIOMA = "glioma"
    MENINGIOMA = "meningioma"
    MENINGIOMA_MULTIMODAL = "meningioma_multimodal"


class PipelineStatus(str, Enum):
    """Execution status of the segmentation pipeline."""
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class RequestMetadata:
    """Session and tracking metadata for an inference request."""
    request_id: str
    tumor_type: TumorType
    patient_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


@dataclass
class InputMetadata:
    """Input MRI acquisition specifications and spatial geometry."""
    modality_names: List[str]
    original_shape: List[int]
    original_spacing: List[float]
    voxel_volume_mm3: float
    original_origin: Optional[List[float]] = None
    original_direction: Optional[List[float]] = None
    original_affine: Optional[List[List[float]]] = None
    resampled_shape: Optional[List[int]] = None
    resampled_spacing: Optional[List[float]] = field(
        default_factory=lambda: [1.0, 1.0, 1.0]
    )


@dataclass
class MaskReference:
    """Location, format, and properties of a segmentation mask volume."""
    mask_path: Optional[str] = None
    mask_format: str = "nifti"  # "nifti", "numpy", "uri", "base64"
    is_original_space: bool = True
    shape: Optional[List[int]] = None
    non_zero_voxels: Optional[int] = None
    checksum_sha256: Optional[str] = None


@dataclass
class VolumetricSummary:
    """Aggregated volumetric measurements across all tumor subregions."""
    total_tumor_voxels: int
    total_tumor_volume_mm3: float
    total_tumor_volume_cm3: float
    voxel_dimensions_mm: List[float]


@dataclass
class GliomaSubregionVolumes:
    """Detailed volumetric breakdown for Glioma subregions (BraTS definitions)."""
    # Whole Tumor (WT = labels 1 + 2 + 3)
    wt_voxels: int
    wt_volume_mm3: float
    wt_volume_cm3: float

    # Tumor Core (TC = labels 1 + 3)
    tc_voxels: int
    tc_volume_mm3: float
    tc_volume_cm3: float

    # Enhancing Tumor (ET = label 3)
    et_voxels: int
    et_volume_mm3: float
    et_volume_cm3: float

    # Non-Enhancing Core / Necrosis (NCR/NET = label 1)
    ncr_net_voxels: Optional[int] = None
    ncr_net_volume_mm3: Optional[float] = None
    ncr_net_volume_cm3: Optional[float] = None

    # Peritumoral Edema (ED = label 2)
    edema_voxels: Optional[int] = None
    edema_volume_mm3: Optional[float] = None
    edema_volume_cm3: Optional[float] = None


@dataclass
class GliomaSubregionMasks:
    """Mask references for individual Glioma subregions and composite masks."""
    multiclass_mask: Optional[MaskReference] = None
    wt_mask: Optional[MaskReference] = None
    tc_mask: Optional[MaskReference] = None
    et_mask: Optional[MaskReference] = None


@dataclass
class GliomaMeasurements:
    """Complete Glioma-specific payload including subregions and masks."""
    subregions: GliomaSubregionVolumes
    masks: Optional[GliomaSubregionMasks] = None
    label_definitions: Dict[int, str] = field(
        default_factory=lambda: {
            0: "Background",
            1: "NCR/NET (Necrotic / Non-Enhancing Core)",
            2: "ED (Peritumoral Edema)",
            3: "ET (Enhancing Tumor)",
        }
    )


@dataclass
class MeningiomaMeasurements:
    """Complete Meningioma-specific payload for Gross Tumor Volume (GTV)."""
    gtv_voxels: int
    gtv_volume_mm3: float
    gtv_volume_cm3: float
    gtv_mask: Optional[MaskReference] = None
    is_original_space_restored: bool = True
    label_definitions: Dict[int, str] = field(
        default_factory=lambda: {
            0: "Background",
            1: "Meningioma GTV (Gross Tumor Volume)",
        }
    )


@dataclass
class MeningiomaMultimodalSubregionVolumes:
    """Detailed volumetric breakdown for Multimodal Meningioma subregions (BraTS definitions)."""
    # Whole Tumor (WT = labels 1 + 2 + 3)
    wt_voxels: int
    wt_volume_mm3: float
    wt_volume_cm3: float

    # Tumor Core (TC = labels 1 + 3)
    tc_voxels: int
    tc_volume_mm3: float
    tc_volume_cm3: float

    # Enhancing Tumor (ET = label 3)
    et_voxels: int
    et_volume_mm3: float
    et_volume_cm3: float

    # Non-Enhancing Core / Necrosis (NCR = label 1)
    ncr_net_voxels: Optional[int] = None
    ncr_net_volume_mm3: Optional[float] = None
    ncr_net_volume_cm3: Optional[float] = None

    # Peritumoral Edema (ED = label 2)
    edema_voxels: Optional[int] = None
    edema_volume_mm3: Optional[float] = None
    edema_volume_cm3: Optional[float] = None


@dataclass
class MeningiomaMultimodalMeasurements:
    """Complete Multimodal Meningioma-specific payload including subregions and masks."""
    subregions: MeningiomaMultimodalSubregionVolumes
    masks: Optional[GliomaSubregionMasks] = None
    label_definitions: Dict[int, str] = field(
        default_factory=lambda: {
            0: "Background",
            1: "Necrotic Tumor Core (NCR)",
            2: "Peritumoral Edema (ED)",
            3: "Enhancing Tumor (ET)",
        }
    )
    is_original_space_restored: bool = True
    disclaimer: str = (
        "Research and prototype tool only. Not for primary diagnostic or clinical decision-making."
    )


@dataclass
class InferenceQualityStatus:
    """Technical and diagnostic quality status of the pipeline run."""
    status: PipelineStatus = PipelineStatus.SUCCESS
    inference_success: bool = True
    pipeline_frozen: bool = True
    checkpoint_used: str = ""
    threshold_applied: float = 0.0
    min_component_size_voxels: int = 0
    warnings: List[str] = field(default_factory=list)
    qc_verdict: Optional[str] = None
    qc_notes: Optional[str] = None


@dataclass
class OfflineEvaluationMetrics:
    """
    Optional evaluation metrics for offline benchmarking against ground truth.
    
    IMPORTANT:
    This section is strictly for offline research/validation scenarios where ground-truth
    annotations are present. It is never populated with fake values for unblinded/unlabeled
    clinical cohorts (e.g. official BraTS Meningioma validation set).
    """
    has_ground_truth: bool = False
    dice: Optional[Dict[str, float]] = None
    iou: Optional[Dict[str, float]] = None
    precision: Optional[Dict[str, float]] = None
    recall: Optional[Dict[str, float]] = None
    notes: Optional[str] = None


@dataclass
class SegmentationResult:
    """
    Master unified result schema for multi-modal brain tumor segmentation.
    
    Encapsulates all request metadata, input geometry, volumetric quantifications,
    pathology-specific subregions, quality status, and optional offline evaluations.
    """
    request: RequestMetadata
    input_metadata: InputMetadata
    volumetrics: VolumetricSummary
    quality_status: InferenceQualityStatus
    glioma: Optional[GliomaMeasurements] = None
    meningioma: Optional[MeningiomaMeasurements] = None
    meningioma_multimodal: Optional[MeningiomaMultimodalMeasurements] = None
    offline_evaluation: Optional[OfflineEvaluationMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result schema instance into a standard JSON-serializable dictionary."""
        def _custom_asdict_factory(data):
            def _convert(obj):
                if isinstance(obj, Enum):
                    return obj.value
                if isinstance(obj, dict):
                    return {k: _convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_convert(v) for v in obj]
                if isinstance(obj, tuple):
                    return [_convert(v) for v in obj]
                return obj
            return _convert(dict(data))

        raw_dict = asdict(self, dict_factory=_custom_asdict_factory)
        return raw_dict

    def to_json(self, indent: int = 2) -> str:
        """Serialize the result schema to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentationResult":
        """Reconstruct a SegmentationResult instance from a dictionary."""
        # Deep reconstruction of sub-dataclasses
        req_data = data.get("request", {})
        req = RequestMetadata(
            request_id=req_data["request_id"],
            tumor_type=TumorType(req_data["tumor_type"]),
            patient_id=req_data.get("patient_id"),
            session_id=req_data.get("session_id"),
            timestamp=req_data.get("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
        )

        inp_data = data.get("input_metadata", {})
        inp = InputMetadata(
            modality_names=inp_data["modality_names"],
            original_shape=inp_data["original_shape"],
            original_spacing=inp_data["original_spacing"],
            voxel_volume_mm3=inp_data["voxel_volume_mm3"],
            original_origin=inp_data.get("original_origin"),
            original_direction=inp_data.get("original_direction"),
            original_affine=inp_data.get("original_affine"),
            resampled_shape=inp_data.get("resampled_shape"),
            resampled_spacing=inp_data.get("resampled_spacing", [1.0, 1.0, 1.0]),
        )

        vol_data = data.get("volumetrics", {})
        vol = VolumetricSummary(
            total_tumor_voxels=vol_data["total_tumor_voxels"],
            total_tumor_volume_mm3=vol_data["total_tumor_volume_mm3"],
            total_tumor_volume_cm3=vol_data["total_tumor_volume_cm3"],
            voxel_dimensions_mm=vol_data["voxel_dimensions_mm"],
        )

        qs_data = data.get("quality_status", {})
        qs = InferenceQualityStatus(
            status=PipelineStatus(qs_data.get("status", "success")),
            inference_success=qs_data.get("inference_success", True),
            pipeline_frozen=qs_data.get("pipeline_frozen", True),
            checkpoint_used=qs_data.get("checkpoint_used", ""),
            threshold_applied=qs_data.get("threshold_applied", 0.0),
            min_component_size_voxels=qs_data.get("min_component_size_voxels", 0),
            warnings=qs_data.get("warnings", []),
            qc_verdict=qs_data.get("qc_verdict"),
            qc_notes=qs_data.get("qc_notes"),
        )

        glioma = None
        if data.get("glioma"):
            g_data = data["glioma"]
            sub_data = g_data["subregions"]
            subregions = GliomaSubregionVolumes(
                wt_voxels=sub_data["wt_voxels"],
                wt_volume_mm3=sub_data["wt_volume_mm3"],
                wt_volume_cm3=sub_data["wt_volume_cm3"],
                tc_voxels=sub_data["tc_voxels"],
                tc_volume_mm3=sub_data["tc_volume_mm3"],
                tc_volume_cm3=sub_data["tc_volume_cm3"],
                et_voxels=sub_data["et_voxels"],
                et_volume_mm3=sub_data["et_volume_mm3"],
                et_volume_cm3=sub_data["et_volume_cm3"],
                ncr_net_voxels=sub_data.get("ncr_net_voxels"),
                ncr_net_volume_mm3=sub_data.get("ncr_net_volume_mm3"),
                ncr_net_volume_cm3=sub_data.get("ncr_net_volume_cm3"),
                edema_voxels=sub_data.get("edema_voxels"),
                edema_volume_mm3=sub_data.get("edema_volume_mm3"),
                edema_volume_cm3=sub_data.get("edema_volume_cm3"),
            )
            masks = None
            if g_data.get("masks"):
                m_data = g_data["masks"]
                def _build_mask_ref(d):
                    if not d:
                        return None
                    return MaskReference(
                        mask_path=d.get("mask_path"),
                        mask_format=d.get("mask_format", "nifti"),
                        is_original_space=d.get("is_original_space", True),
                        shape=d.get("shape"),
                        non_zero_voxels=d.get("non_zero_voxels"),
                        checksum_sha256=d.get("checksum_sha256"),
                    )
                masks = GliomaSubregionMasks(
                    multiclass_mask=_build_mask_ref(m_data.get("multiclass_mask")),
                    wt_mask=_build_mask_ref(m_data.get("wt_mask")),
                    tc_mask=_build_mask_ref(m_data.get("tc_mask")),
                    et_mask=_build_mask_ref(m_data.get("et_mask")),
                )
            raw_labels = g_data.get("label_definitions")
            label_defs = {int(k): v for k, v in raw_labels.items()} if raw_labels else {
                0: "Background",
                1: "NCR/NET (Necrotic / Non-Enhancing Core)",
                2: "ED (Peritumoral Edema)",
                3: "ET (Enhancing Tumor)",
            }
            glioma = GliomaMeasurements(
                subregions=subregions,
                masks=masks,
                label_definitions=label_defs,
            )

        meningioma = None
        if data.get("meningioma"):
            m_data = data["meningioma"]
            gtv_mask = None
            if m_data.get("gtv_mask"):
                gm_data = m_data["gtv_mask"]
                gtv_mask = MaskReference(
                    mask_path=gm_data.get("mask_path"),
                    mask_format=gm_data.get("mask_format", "nifti"),
                    is_original_space=gm_data.get("is_original_space", True),
                    shape=gm_data.get("shape"),
                    non_zero_voxels=gm_data.get("non_zero_voxels"),
                    checksum_sha256=gm_data.get("checksum_sha256"),
                )
            raw_labels = m_data.get("label_definitions")
            label_defs = {int(k): v for k, v in raw_labels.items()} if raw_labels else {
                0: "Background",
                1: "Meningioma GTV (Gross Tumor Volume)",
            }
            meningioma = MeningiomaMeasurements(
                gtv_voxels=m_data["gtv_voxels"],
                gtv_volume_mm3=m_data["gtv_volume_mm3"],
                gtv_volume_cm3=m_data["gtv_volume_cm3"],
                gtv_mask=gtv_mask,
                is_original_space_restored=m_data.get("is_original_space_restored", True),
                label_definitions=label_defs,
            )

        meningioma_multimodal = None
        if data.get("meningioma_multimodal"):
            mm_data = data["meningioma_multimodal"]
            sub_data = mm_data["subregions"]
            subregions = MeningiomaMultimodalSubregionVolumes(
                wt_voxels=sub_data["wt_voxels"],
                wt_volume_mm3=sub_data["wt_volume_mm3"],
                wt_volume_cm3=sub_data["wt_volume_cm3"],
                tc_voxels=sub_data["tc_voxels"],
                tc_volume_mm3=sub_data["tc_volume_mm3"],
                tc_volume_cm3=sub_data["tc_volume_cm3"],
                et_voxels=sub_data["et_voxels"],
                et_volume_mm3=sub_data["et_volume_mm3"],
                et_volume_cm3=sub_data["et_volume_cm3"],
                ncr_net_voxels=sub_data.get("ncr_net_voxels"),
                ncr_net_volume_mm3=sub_data.get("ncr_net_volume_mm3"),
                ncr_net_volume_cm3=sub_data.get("ncr_net_volume_cm3"),
                edema_voxels=sub_data.get("edema_voxels"),
                edema_volume_mm3=sub_data.get("edema_volume_mm3"),
                edema_volume_cm3=sub_data.get("edema_volume_cm3"),
            )
            masks = None
            if mm_data.get("masks"):
                m_data = mm_data["masks"]
                def _build_mask_ref(d):
                    if not d:
                        return None
                    return MaskReference(
                        mask_path=d.get("mask_path"),
                        mask_format=d.get("mask_format", "nifti"),
                        is_original_space=d.get("is_original_space", True),
                        shape=d.get("shape"),
                        non_zero_voxels=d.get("non_zero_voxels"),
                        checksum_sha256=d.get("checksum_sha256"),
                    )
                masks = GliomaSubregionMasks(
                    multiclass_mask=_build_mask_ref(m_data.get("multiclass_mask")),
                    wt_mask=_build_mask_ref(m_data.get("wt_mask")),
                    tc_mask=_build_mask_ref(m_data.get("tc_mask")),
                    et_mask=_build_mask_ref(m_data.get("et_mask")),
                )
            raw_labels = mm_data.get("label_definitions")
            label_defs = {int(k): v for k, v in raw_labels.items()} if raw_labels else {
                0: "Background",
                1: "Necrotic Tumor Core (NCR)",
                2: "Peritumoral Edema (ED)",
                3: "Enhancing Tumor (ET)",
            }
            meningioma_multimodal = MeningiomaMultimodalMeasurements(
                subregions=subregions,
                masks=masks,
                label_definitions=label_defs,
                is_original_space_restored=mm_data.get("is_original_space_restored", True),
                disclaimer=mm_data.get(
                    "disclaimer",
                    "Research and prototype tool only. Not for standalone clinical diagnosis or medical decision making."
                ),
            )

        offline_eval = None
        if data.get("offline_evaluation"):
            oe_data = data["offline_evaluation"]
            offline_eval = OfflineEvaluationMetrics(
                has_ground_truth=oe_data.get("has_ground_truth", False),
                dice=oe_data.get("dice"),
                iou=oe_data.get("iou"),
                precision=oe_data.get("precision"),
                recall=oe_data.get("recall"),
                notes=oe_data.get("notes"),
            )

        return cls(
            request=req,
            input_metadata=inp,
            volumetrics=vol,
            quality_status=qs,
            glioma=glioma,
            meningioma=meningioma,
            meningioma_multimodal=meningioma_multimodal,
            offline_evaluation=offline_eval,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SegmentationResult":
        """Reconstruct a SegmentationResult instance from a JSON string."""
        return cls.from_dict(json.loads(json_str))


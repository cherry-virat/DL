"""
Isolated Unit Tests for Common Result Schema.
=============================================

Tests:
1. Valid Glioma result (WT, TC, ET, volumes, mask metadata, spatial metadata).
2. Valid Meningioma result (GTV, volume, mask metadata, spatial metadata).
3. Serialization to JSON.
4. Deserialization from JSON (round-trip equality).
5. Validation failure for invalid tumor type.
6. Validation failure for missing required fields.
7. Confirmation that evaluation metrics are optional and Meningioma inference
   does not require Dice/IoU.
"""

import json
import pytest
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


def test_valid_glioma_result():
    """Test 1: Valid Glioma result with WT, TC, ET, volumes, mask and spatial metadata."""
    req = RequestMetadata(
        request_id="REQ-GLI-001",
        tumor_type=TumorType.GLIOMA,
        patient_id="BraTS-GLI-01666-000",
        session_id="SESS-001",
    )
    inp = InputMetadata(
        modality_names=["T1", "T1c", "T2", "FLAIR"],
        original_shape=[240, 240, 155],
        original_spacing=[1.0, 1.0, 1.0],
        voxel_volume_mm3=1.0,
        original_origin=[0.0, 0.0, 0.0],
        original_direction=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        resampled_shape=[240, 240, 155],
        resampled_spacing=[1.0, 1.0, 1.0],
    )
    subregions = GliomaSubregionVolumes(
        wt_voxels=114532,
        wt_volume_mm3=114532.0,
        wt_volume_cm3=114.532,
        tc_voxels=10147,
        tc_volume_mm3=10147.0,
        tc_volume_cm3=10.147,
        et_voxels=2396,
        et_volume_mm3=2396.0,
        et_volume_cm3=2.396,
        ncr_net_voxels=7751,
        ncr_net_volume_mm3=7751.0,
        ncr_net_volume_cm3=7.751,
        edema_voxels=104385,
        edema_volume_mm3=104385.0,
        edema_volume_cm3=104.385,
    )
    masks = GliomaSubregionMasks(
        multiclass_mask=MaskReference(
            mask_path=None,  # Placeholder or in-memory reference
            mask_format="nifti",
            is_original_space=True,
            shape=[240, 240, 155],
            non_zero_voxels=114532,
        ),
        wt_mask=None,
        tc_mask=None,
        et_mask=None,
    )
    vol = VolumetricSummary(
        total_tumor_voxels=114532,
        total_tumor_volume_mm3=114532.0,
        total_tumor_volume_cm3=114.532,
        voxel_dimensions_mm=[1.0, 1.0, 1.0],
    )
    qs = InferenceQualityStatus(
        status=PipelineStatus.SUCCESS,
        inference_success=True,
        pipeline_frozen=True,
        checkpoint_used="outputs/glioma_training/glioma_best_medium_experiment.pth",
        threshold_applied=0.60,
        min_component_size_voxels=1000,
        qc_verdict="PASS WITH DIFFICULT CASES",
    )
    # Optional offline evaluation using actual formal 20-case aggregate metrics
    offline_eval = OfflineEvaluationMetrics(
        has_ground_truth=True,
        dice={"WT": 0.904232, "TC": 0.766601, "ET": 0.840123},
        iou={"WT": 0.825203},
        precision={"WT": 0.906649},
        recall={"WT": 0.901828},
        notes="Offline formal evaluation metrics from frozen 20-case Glioma validation evaluation, not live inference metrics.",
    )
    result = SegmentationResult(
        request=req,
        input_metadata=inp,
        volumetrics=vol,
        quality_status=qs,
        glioma=GliomaMeasurements(subregions=subregions, masks=masks),
        offline_evaluation=offline_eval,
    )

    assert result.request.tumor_type == TumorType.GLIOMA
    assert result.glioma is not None
    assert result.glioma.subregions.wt_voxels == 114532
    assert result.glioma.subregions.tc_voxels == 10147
    assert result.glioma.subregions.et_voxels == 2396
    assert result.glioma.subregions.wt_volume_cm3 == 114.532
    assert result.glioma.masks.multiclass_mask.is_original_space is True
    assert result.input_metadata.modality_names == ["T1", "T1c", "T2", "FLAIR"]
    assert result.offline_evaluation.dice["WT"] == 0.904232


def test_valid_meningioma_result():
    """Test 2: Valid Meningioma result with GTV, volume, mask and spatial metadata."""
    req = RequestMetadata(
        request_id="REQ-MEN-001",
        tumor_type=TumorType.MENINGIOMA,
        patient_id="BraTS-MEN-RT-0010-1",
        session_id="SESS-002",
    )
    inp = InputMetadata(
        modality_names=["T1c"],
        original_shape=[512, 512, 176],
        original_spacing=[0.4883, 0.4883, 1.0],
        voxel_volume_mm3=0.23843689,
        original_origin=[-120.0, -110.5, -45.2],
        original_direction=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        resampled_shape=[240, 240, 155],
    )
    gtv_mask = MaskReference(
        mask_path="outputs/meningioma_validation_original_space/BraTS-MEN-RT-0010-1/BraTS-MEN-RT-0010-1_meningioma_prediction.nii.gz",
        mask_format="nifti",
        is_original_space=True,
        shape=[512, 512, 176],
        non_zero_voxels=78650,
    )
    men_meas = MeningiomaMeasurements(
        gtv_voxels=78650,
        gtv_volume_mm3=18753.06,
        gtv_volume_cm3=18.75,
        gtv_mask=gtv_mask,
        is_original_space_restored=True,
    )
    vol = VolumetricSummary(
        total_tumor_voxels=78650,
        total_tumor_volume_mm3=18753.06,
        total_tumor_volume_cm3=18.75,
        voxel_dimensions_mm=[0.4883, 0.4883, 1.0],
    )
    qs = InferenceQualityStatus(
        status=PipelineStatus.SUCCESS,
        inference_success=True,
        pipeline_frozen=True,
        checkpoint_used="outputs/meningioma_training/meningioma_best.pth",
        threshold_applied=0.125,
        min_component_size_voxels=2500,
        qc_verdict="PASS WITH UNCERTAINTY",
    )
    result = SegmentationResult(
        request=req,
        input_metadata=inp,
        volumetrics=vol,
        quality_status=qs,
        meningioma=men_meas,
    )

    assert result.request.tumor_type == TumorType.MENINGIOMA
    assert result.meningioma is not None
    assert result.meningioma.gtv_voxels == 78650
    assert result.meningioma.gtv_volume_cm3 == 18.75
    assert result.meningioma.is_original_space_restored is True
    assert result.volumetrics.total_tumor_volume_cm3 == 18.75
    assert result.input_metadata.original_shape == [512, 512, 176]


def test_serialization_to_json():
    """Test 3: Serialization to standard JSON."""
    req = RequestMetadata(request_id="REQ-TEST-003", tumor_type=TumorType.GLIOMA)
    inp = InputMetadata(
        modality_names=["T1c"],
        original_shape=[240, 240, 155],
        original_spacing=[1.0, 1.0, 1.0],
        voxel_volume_mm3=1.0,
    )
    vol = VolumetricSummary(100, 100.0, 0.1, [1.0, 1.0, 1.0])
    qs = InferenceQualityStatus()
    result = SegmentationResult(request=req, input_metadata=inp, volumetrics=vol, quality_status=qs)

    json_str = result.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["request"]["request_id"] == "REQ-TEST-003"
    assert parsed["request"]["tumor_type"] == "glioma"
    assert parsed["volumetrics"]["total_tumor_volume_cm3"] == 0.1


def test_deserialization_from_json():
    """Test 4: Deserialization from JSON and round-trip fidelity."""
    sample_data = {
        "request": {
            "request_id": "REQ-ROUNDTRIP",
            "tumor_type": "meningioma",
            "patient_id": "PT-99",
            "session_id": "S-99",
            "timestamp": "2026-10-08T05:00:00Z"
        },
        "input_metadata": {
            "modality_names": ["T1c"],
            "original_shape": [256, 256, 160],
            "original_spacing": [0.9375, 0.9375, 1.0],
            "voxel_volume_mm3": 0.8789,
            "resampled_shape": [240, 240, 155],
            "resampled_spacing": [1.0, 1.0, 1.0]
        },
        "volumetrics": {
            "total_tumor_voxels": 12000,
            "total_tumor_volume_mm3": 10546.8,
            "total_tumor_volume_cm3": 10.55,
            "voxel_dimensions_mm": [0.9375, 0.9375, 1.0]
        },
        "quality_status": {
            "status": "success",
            "inference_success": True,
            "pipeline_frozen": True,
            "checkpoint_used": "outputs/meningioma_training/meningioma_best.pth",
            "threshold_applied": 0.125,
            "min_component_size_voxels": 2500,
            "warnings": [],
            "qc_verdict": "PASS WITH UNCERTAINTY",
            "qc_notes": None
        },
        "glioma": None,
        "meningioma": {
            "gtv_voxels": 12000,
            "gtv_volume_mm3": 10546.8,
            "gtv_volume_cm3": 10.55,
            "gtv_mask": {
                "mask_path": None,
                "mask_format": "nifti",
                "is_original_space": True,
                "shape": [256, 256, 160],
                "non_zero_voxels": 12000,
                "checksum_sha256": None
            },
            "is_original_space_restored": True,
            "label_definitions": {"0": "Background", "1": "Meningioma GTV"}
        },
        "offline_evaluation": None
    }

    json_str = json.dumps(sample_data)
    result = SegmentationResult.from_json(json_str)

    assert result.request.request_id == "REQ-ROUNDTRIP"
    assert result.request.tumor_type == TumorType.MENINGIOMA
    assert result.meningioma.gtv_volume_cm3 == 10.55
    assert result.meningioma.gtv_mask.shape == [256, 256, 160]
    assert result.volumetrics.total_tumor_voxels == 12000

    # Ensure re-serialization is consistent
    reserialized = json.loads(result.to_json())
    assert reserialized["request"]["request_id"] == "REQ-ROUNDTRIP"
    assert reserialized["meningioma"]["gtv_voxels"] == 12000


def test_invalid_tumor_type_failure():
    """Test 5: Validation failure when invalid tumor type is provided."""
    with pytest.raises(ValueError):
        TumorType("invalid_pathology")

    with pytest.raises(ValueError):
        RequestMetadata(request_id="REQ-ERR", tumor_type=TumorType("sarcoma"))

    # Test failure during dictionary deserialization
    invalid_data = {
        "request": {
            "request_id": "REQ-ERR",
            "tumor_type": "unknown_tumor"
        }
    }
    with pytest.raises(ValueError):
        SegmentationResult.from_dict(invalid_data)


def test_missing_required_fields_failure():
    """Test 6: Validation failure when required fields are missing."""
    # Missing required argument 'tumor_type' in RequestMetadata
    with pytest.raises(TypeError):
        RequestMetadata(request_id="REQ-MISSING")  # pylint: disable=no-value-for-parameter

    # Missing required spatial fields in InputMetadata
    with pytest.raises(TypeError):
        InputMetadata(modality_names=["T1c"])  # pylint: disable=no-value-for-parameter

    # Missing required subregion fields in GliomaSubregionVolumes
    with pytest.raises(TypeError):
        GliomaSubregionVolumes(wt_voxels=100)  # pylint: disable=no-value-for-parameter

    # Missing root dataclass required arguments
    with pytest.raises(TypeError):
        SegmentationResult()  # pylint: disable=no-value-for-parameter


def test_optional_offline_evaluation_and_meningioma_no_dice():
    """Test 7: Confirm evaluation metrics are optional and Meningioma does not require Dice/IoU."""
    # Clinical inference result with NO evaluation metadata
    req = RequestMetadata(request_id="REQ-CLINICAL", tumor_type=TumorType.MENINGIOMA)
    inp = InputMetadata(["T1c"], [512, 512, 176], [0.5, 0.5, 1.0], 0.25)
    vol = VolumetricSummary(5000, 1250.0, 1.25, [0.5, 0.5, 1.0])
    qs = InferenceQualityStatus(qc_verdict="PASS WITH UNCERTAINTY")
    men_meas = MeningiomaMeasurements(gtv_voxels=5000, gtv_volume_mm3=1250.0, gtv_volume_cm3=1.25)

    result = SegmentationResult(
        request=req,
        input_metadata=inp,
        volumetrics=vol,
        quality_status=qs,
        meningioma=men_meas,
        offline_evaluation=None,  # Not provided during clinical inference
    )

    # Validate that offline_evaluation is None and serialization succeeds
    res_dict = result.to_dict()
    assert res_dict["offline_evaluation"] is None
    assert "dice" not in res_dict or res_dict.get("offline_evaluation") is None

    # When offline_evaluation is explicitly initialized for Meningioma validation, has_ground_truth=False and dice=None
    eval_no_gt = OfflineEvaluationMetrics(
        has_ground_truth=False,
        dice=None,
        iou=None,
        notes="Meningioma validation set has no GTV ground truth. Dice/IoU unavailable.",
    )
    result.offline_evaluation = eval_no_gt
    res_dict_eval = result.to_dict()
    assert res_dict_eval["offline_evaluation"]["has_ground_truth"] is False
    assert res_dict_eval["offline_evaluation"]["dice"] is None
    assert res_dict_eval["offline_evaluation"]["iou"] is None

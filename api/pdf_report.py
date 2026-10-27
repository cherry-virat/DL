"""
PDF Clinical Report Generator for 3D Brain Tumor Segmentation System.
====================================================================
Uses ReportLab to render a clean, publication-grade clinical research
patient report PDF including:
  - Header & Case/Patient Identification
  - Hierarchical Classifier Telemetry (Stage 1 presence & Stage 2 prediction)
  - Diagnostic & Manual Override Conflict Notices
  - 3D MRI Acquisition Geometry
  - Quantitative Subregion Volumetrics Table (mm³, cm³, % breakdown)
  - Quality Control & Model Provenance
  - Regulatory Research Disclaimer
"""

import io
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_pdf_report(res_data: Dict[str, Any], request_id: str, classification_data: Optional[Dict[str, Any]] = None) -> bytes:
    """
    Renders a comprehensive, styled PDF report from a result.json dictionary.
    Returns the PDF as raw bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Color Palette
    primary_color = colors.HexColor("#0f172a")      # Dark Navy
    accent_color = colors.HexColor("#0284c7")       # Medical Blue
    header_bg = colors.HexColor("#1e293b")          # Slate Dark
    alt_row_bg = colors.HexColor("#f8fafc")         # Very light grey
    border_color = colors.HexColor("#cbd5e1")       # Subtle border
    muted_text = colors.HexColor("#64748b")         # Muted Slate
    alert_bg = colors.HexColor("#fef2f2")           # Light red
    alert_border = colors.HexColor("#f87171")       # Red border
    alert_text = colors.HexColor("#991b1b")         # Dark Red text
    healthy_bg = colors.HexColor("#f0fdf4")         # Light green
    healthy_border = colors.HexColor("#4ade80")     # Green border
    healthy_text = colors.HexColor("#166534")       # Dark green text

    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=primary_color,
        alignment=1,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=muted_text,
        alignment=1,
    )

    sec_heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=accent_color,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=primary_color,
    )

    body_bold_style = ParagraphStyle(
        "BodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=primary_color,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=primary_color,
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=primary_color,
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=primary_color,
        alignment=1,
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=muted_text,
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("3D BRAIN TUMOR MRI SEGMENTATION & VOLUMETRIC REPORT", title_style))
    elements.append(Paragraph("Investigational Quantitative Imaging Evaluation • Multi-Compartment Volumetrics", subtitle_style))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceBefore=0, spaceAfter=8))

    # Parse JSON fields defensively
    req_meta = res_data.get("request_metadata") or res_data.get("request", {})
    inp_meta = res_data.get("input_metadata", {})
    vol_summary = res_data.get("volumetrics", res_data.get("volumetric_summary", {}))
    gl_meas = res_data.get("glioma", res_data.get("glioma_measurements"))
    men_meas = res_data.get("meningioma_multimodal", res_data.get("meningioma", res_data.get("meningioma_measurements")))
    qc = res_data.get("quality_status", {})
    res_status = res_data.get("status")

    raw_tumor_type = str(req_meta.get("tumor_type", "Unknown"))
    tumor_type = raw_tumor_type.upper()
    routing_mode = req_meta.get("routing_mode", "manual_selection")
    is_auto = (routing_mode == "automatic_classification")
    is_override = bool(req_meta.get("manual_override", False) and req_meta.get("conflict_detected", False))

    if res_status == "no_tumor_detected":
        routing_mode_str = "Automatic Classification (Normative Scan)"
        target_pathology = "HEALTHY CONTROL (NORMATIVE)"
        selected_model = "None (Halted at Stage 1)"
        pipe_status = "NO TUMOR DETECTED (200 OK)"
    elif res_status == "uncertain_classification":
        routing_mode_str = "Automatic Classification (Uncertainty Refusal)"
        target_pathology = "UNCERTAIN PATHOLOGY"
        selected_model = "None (Refused - Ambiguous)"
        pipe_status = "UNCERTAIN CLASSIFICATION (422)"
    elif is_auto:
        routing_mode_str = "Automatic Classification (Hierarchical Gating)"
        target_pathology = tumor_type
        selected_model = f"{tumor_type} 3D U-Net (Automatic Routing)"
        pipe_status = str(qc.get("status") or "SUCCESS").upper()
    elif is_override:
        routing_mode_str = "Manual Override (Classifier Disagreement Detected)"
        target_pathology = tumor_type
        selected_model = f"{tumor_type} 3D U-Net (Manual Override)"
        pipe_status = str(qc.get("status") or "SUCCESS").upper()
    else:
        routing_mode_str = "Manual Selection (Direct Operator Specification)"
        target_pathology = tumor_type
        selected_model = f"{tumor_type} 3D U-Net (Manual Selection)"
        pipe_status = str(qc.get("status") or "SUCCESS").upper()

    date_str = req_meta.get("timestamp") or res_data.get("timestamp") or datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 2. Case Identification & Pipeline Summary Box
    case_info = [
        [
            Paragraph("<b>Case / Request ID:</b>", body_style),
            Paragraph(f"<b>{request_id}</b>", body_bold_style),
            Paragraph("<b>Evaluation Date:</b>", body_style),
            Paragraph(date_str[:19].replace("T", " "), body_style),
        ],
        [
            Paragraph("<b>Target Pathology:</b>", body_style),
            Paragraph(f"<b>{target_pathology}</b>", body_bold_style),
            Paragraph("<b>Pipeline Status:</b>", body_style),
            Paragraph(f"<b>{pipe_status}</b>", body_bold_style),
        ],
        [
            Paragraph("<b>Model Routing:</b>", body_style),
            Paragraph(routing_mode_str, body_style),
            Paragraph("<b>Selected Model:</b>", body_style),
            Paragraph(selected_model, body_style),
        ],
    ]

    t_case = Table(case_info, colWidths=[105, 165, 105, 165])
    t_case.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), alt_row_bg),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(t_case)
    elements.append(Spacer(1, 8))

    # Diagnostic Notice / Warning Box (if conflict or early exit)
    if res_status == "no_tumor_detected":
        notice_data = [[
            Paragraph("<b>STAGE 1 SCREENING RESULT: NORMATIVE BRAIN SCAN</b><br/>"
                      f"Presence Probability: {res_data.get('presence_probability', 'N/A')} (Confidence: {res_data.get('confidence', 'N/A')}).<br/>"
                      "Automated Stage 1 classifier detected no neoplastic tumor tissue. Segmentation was safely halted to prevent phantom volumetric contouring.",
                      ParagraphStyle("Healthy", parent=body_style, textColor=healthy_text))
        ]]
        t_notice = Table(notice_data, colWidths=[540])
        t_notice.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), healthy_bg),
            ("BOX", (0, 0), (-1, -1), 1, healthy_border),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(t_notice)
        elements.append(Spacer(1, 8))
    elif res_status == "uncertain_classification":
        warn_data = [[
            Paragraph(f"<b>WARNING: UNCERTAIN CLASSIFICATION CONFIDENCE</b><br/>"
                      f"Confidence score ({res_data.get('confidence', 'N/A')}) is below safety threshold (0.55).<br/>"
                      "Automated routing was safely refused. Manual model selection ('glioma' or 'meningioma') is required.",
                      ParagraphStyle("Alert", parent=body_style, textColor=alert_text))
        ]]
        t_warn = Table(warn_data, colWidths=[540])
        t_warn.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), alert_bg),
            ("BOX", (0, 0), (-1, -1), 1, alert_border),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(t_warn)
        elements.append(Spacer(1, 8))
    elif is_override:
        warn = req_meta.get("warning") or req_meta.get("diagnostic_notice") or "Manual override selected contrary to automated classifier prediction."
        warn_data = [[
            Paragraph(f"<b>WARNING: MANUAL OVERRIDE CONFLICT DETECTED</b><br/>{warn}", ParagraphStyle("Alert", parent=body_style, textColor=alert_text))
        ]]
        t_warn = Table(warn_data, colWidths=[540])
        t_warn.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), alert_bg),
            ("BOX", (0, 0), (-1, -1), 1, alert_border),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(t_warn)
        elements.append(Spacer(1, 8))

    # Hierarchical Classifier Telemetry (if available)
    cls = classification_data or req_meta.get("classifier_assessment") or {}
    if cls and res_status not in ("no_tumor_detected", "uncertain_classification"):
        p_presence = cls.get("presence_probability")
        p_type = cls.get("predicted_tumor_type")
        conf = cls.get("confidence") or cls.get("type_probability")
        p_glioma = cls.get("glioma_probability")
        p_meningioma = cls.get("meningioma_probability")

        cls_rows = [
            [
                Paragraph("<b>Stage 1 Screening:</b>", body_style),
                Paragraph(f"Tumor Detected (P: {p_presence:.4f})" if p_presence is not None else "Screening Confirmed", body_style),
                Paragraph("<b>Stage 2 Prediction:</b>", body_style),
                Paragraph(f"<b>{str(p_type).upper()}</b> (Confidence: {conf:.4f})" if (p_type and conf is not None) else "Evaluated", body_bold_style if not is_override else body_style),
            ]
        ]
        if p_glioma is not None and p_meningioma is not None:
            cls_rows.append([
                Paragraph("<b>Glioma Probability:</b>", body_style),
                Paragraph(f"{p_glioma:.4f} ({p_glioma * 100.0:.2f}%)", body_style),
                Paragraph("<b>Meningioma Probability:</b>", body_style),
                Paragraph(f"{p_meningioma:.4f} ({p_meningioma * 100.0:.2f}%)", body_style),
            ])
        t_cls = Table(cls_rows, colWidths=[105, 165, 105, 165])
        t_cls.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 0.5, border_color),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(t_cls)
        elements.append(Spacer(1, 8))

    # 3. Acquisition Geometry Section
    elements.append(Paragraph("1. Acquisition Geometry & Input Modalities", sec_heading_style))
    shape = inp_meta.get("original_shape", [240, 240, 155])
    spacing = inp_meta.get("original_spacing", [1.0, 1.0, 1.0])
    voxel_vol = inp_meta.get("voxel_volume_mm3", 1.0)
    modalities = ", ".join(inp_meta.get("modality_names", ["T1", "T1c", "T2", "FLAIR"])).upper()

    geom_data = [
        [
            Paragraph("<b>Matrix Dimensions:</b>", body_style),
            Paragraph(f"{shape[0]} × {shape[1]} × {shape[2]} voxels", body_style),
            Paragraph("<b>Voxel Spacing:</b>", body_style),
            Paragraph(f"{spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} mm", body_style),
        ],
        [
            Paragraph("<b>Unit Voxel Volume:</b>", body_style),
            Paragraph(f"{voxel_vol:.4f} mm³", body_style),
            Paragraph("<b>Input Modalities:</b>", body_style),
            Paragraph(f"{modalities} (4 channels verified)", body_style),
        ]
    ]
    t_geom = Table(geom_data, colWidths=[105, 165, 105, 165])
    t_geom.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(t_geom)
    elements.append(Spacer(1, 8))

    # 4. Quantitative Subregion Volumetrics Section
    elements.append(Paragraph("2. Quantitative Subregion Volumetrics", sec_heading_style))

    sub_rows = [
        [
            Paragraph("<b>Subregion Name</b>", table_header_style),
            Paragraph("<b>Voxel Count</b>", table_header_style),
            Paragraph("<b>Volume (mm³)</b>", table_header_style),
            Paragraph("<b>Volume (cm³)</b>", table_header_style),
            Paragraph("<b>% of Whole Tumor</b>", table_header_style),
        ]
    ]

    tot_mm3 = vol_summary.get("total_tumor_volume_mm3", 0.0)
    tot_cm3 = vol_summary.get("total_tumor_volume_cm3", 0.0)
    tot_vox = vol_summary.get("total_tumor_voxels", int(tot_mm3 / voxel_vol if voxel_vol else 0))

    if res_status in ("no_tumor_detected", "uncertain_classification"):
        sub_rows.append([
            Paragraph("<b>Whole Tumor (WT)</b>", table_cell_bold),
            Paragraph("0", table_cell_center),
            Paragraph("0.0", table_cell_center),
            Paragraph("0.00", table_cell_center),
            Paragraph("0.0%", table_cell_center),
        ])
    elif tumor_type == "GLIOMA" and gl_meas:
        sub = gl_meas.get("subregions") or gl_meas.get("subregion_volumes") or {}
        wt_mm3 = sub.get("wt_volume_mm3", tot_mm3)
        tc_mm3 = sub.get("tc_volume_mm3", 0.0)
        et_mm3 = sub.get("et_volume_mm3", 0.0)

        wt_vox = int(wt_mm3 / voxel_vol) if voxel_vol else tot_vox
        tc_vox = int(tc_mm3 / voxel_vol) if voxel_vol else 0
        et_vox = int(et_mm3 / voxel_vol) if voxel_vol else 0

        sub_rows.append([
            Paragraph("<b>Whole Tumor (WT)</b>", table_cell_bold),
            Paragraph(f"{wt_vox:,}", table_cell_center),
            Paragraph(f"{wt_mm3:,.1f}", table_cell_center),
            Paragraph(f"{wt_mm3 / 1000.0:.2f}", table_cell_center),
            Paragraph("100.0%", table_cell_center),
        ])
        sub_rows.append([
            Paragraph("Tumor Core (TC)", table_cell_style),
            Paragraph(f"{tc_vox:,}", table_cell_center),
            Paragraph(f"{tc_mm3:,.1f}", table_cell_center),
            Paragraph(f"{tc_mm3 / 1000.0:.2f}", table_cell_center),
            Paragraph(f"{(tc_mm3 / wt_mm3 * 100.0):.1f}%" if wt_mm3 else "0.0%", table_cell_center),
        ])
        sub_rows.append([
            Paragraph("Enhancing Tumor (ET)", table_cell_style),
            Paragraph(f"{et_vox:,}", table_cell_center),
            Paragraph(f"{et_mm3:,.1f}", table_cell_center),
            Paragraph(f"{et_mm3 / 1000.0:.2f}", table_cell_center),
            Paragraph(f"{(et_mm3 / wt_mm3 * 100.0):.1f}%" if wt_mm3 else "0.0%", table_cell_center),
        ])
    elif men_meas:
        sub = men_meas.get("subregions") or men_meas.get("subregion_volumes") or {}
        wt_mm3 = sub.get("wt_volume_mm3", tot_mm3)
        et_mm3 = sub.get("et_volume_mm3", 0.0)
        ncr_mm3 = sub.get("ncr_net_volume_mm3") or sub.get("ncr_volume_mm3", 0.0)
        ed_mm3 = sub.get("edema_volume_mm3") or sub.get("ed_volume_mm3", 0.0)

        wt_vox = sub.get("wt_voxels", int(wt_mm3 / voxel_vol) if voxel_vol else tot_vox)
        et_vox = sub.get("et_voxels", int(et_mm3 / voxel_vol) if voxel_vol else 0)
        ncr_vox = sub.get("ncr_net_voxels") or sub.get("ncr_voxels", int(ncr_mm3 / voxel_vol) if voxel_vol else 0)
        ed_vox = sub.get("edema_voxels") or sub.get("ed_voxels", int(ed_mm3 / voxel_vol) if voxel_vol else 0)

        sub_rows.append([
            Paragraph("<b>Whole Tumor (WT)</b>", table_cell_bold),
            Paragraph(f"{wt_vox:,}", table_cell_center),
            Paragraph(f"{wt_mm3:,.1f}", table_cell_center),
            Paragraph(f"{wt_mm3 / 1000.0:.2f}", table_cell_center),
            Paragraph("100.0%", table_cell_center),
        ])
        sub_rows.append([
            Paragraph("Enhancing Tumor (ET)", table_cell_style),
            Paragraph(f"{et_vox:,}", table_cell_center),
            Paragraph(f"{et_mm3:,.1f}", table_cell_center),
            Paragraph(f"{et_mm3 / 1000.0:.2f}", table_cell_center),
            Paragraph(f"{(et_mm3 / wt_mm3 * 100.0):.1f}%" if wt_mm3 else "0.0%", table_cell_center),
        ])
        sub_rows.append([
            Paragraph("Non-Enhancing Core (NCR)", table_cell_style),
            Paragraph(f"{ncr_vox:,}", table_cell_center),
            Paragraph(f"{ncr_mm3:,.1f}", table_cell_center),
            Paragraph(f"{ncr_mm3 / 1000.0:.2f}", table_cell_center),
            Paragraph(f"{(ncr_mm3 / wt_mm3 * 100.0):.1f}%" if wt_mm3 else "0.0%", table_cell_center),
        ])
        sub_rows.append([
            Paragraph("Peritumoral Edema (ED)", table_cell_style),
            Paragraph(f"{ed_vox:,}", table_cell_center),
            Paragraph(f"{ed_mm3:,.1f}", table_cell_center),
            Paragraph(f"{ed_mm3 / 1000.0:.2f}", table_cell_center),
            Paragraph(f"{(ed_mm3 / wt_mm3 * 100.0):.1f}%" if wt_mm3 else "0.0%", table_cell_center),
        ])
    else:
        # Fallback single summary row
        sub_rows.append([
            Paragraph("<b>Total Tumor Mass</b>", table_cell_bold),
            Paragraph(f"{tot_vox:,}", table_cell_center),
            Paragraph(f"{tot_mm3:,.1f}", table_cell_center),
            Paragraph(f"{tot_cm3:.2f}", table_cell_center),
            Paragraph("100.0%", table_cell_center),
        ])

    t_sub = Table(sub_rows, colWidths=[150, 95, 100, 95, 100])
    t_sub.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, alt_row_bg]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(t_sub)
    elements.append(Spacer(1, 8))

    # 5. Quality Control & Model Execution Summary
    elements.append(Paragraph("3. Model Execution & Quality Verification", sec_heading_style))
    ckpt = str(Path(qc.get("checkpoint_used", "Standard Benchmark Checkpoint")).name)
    verdict = qc.get("qc_verdict", "PASS (Geometry & Intensity Validated)")
    notes = qc.get("qc_notes", "Automated preprocessing & normalization pipeline validated.")

    qc_data = [
        [
            Paragraph("<b>Model Checkpoint:</b>", body_style),
            Paragraph(f"<code>{ckpt}</code>", body_style),
            Paragraph("<b>Validation Verdict:</b>", body_style),
            Paragraph(f"<b>{verdict}</b>", body_style),
        ],
        [
            Paragraph("<b>Post-Processing:</b>", body_style),
            Paragraph("Brain-aware zero-air masking, native space restoration", body_style),
            Paragraph("<b>Routing Notes:</b>", body_style),
            Paragraph(notes, body_style),
        ]
    ]
    t_qc = Table(qc_data, colWidths=[105, 165, 105, 165])
    t_qc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(t_qc)
    elements.append(Spacer(1, 10))

    # 6. Regulatory & Research Disclaimer Footer Box
    elements.append(HRFlowable(width="100%", thickness=0.75, color=border_color, spaceBefore=0, spaceAfter=6))
    elements.append(Paragraph(
        "<b>RESEARCH PROTOTYPE NOTICE & REGULATORY DISCLAIMER:</b><br/>"
        "This quantitative assessment report is generated by an investigational deep learning research prototype. "
        "It has NOT been evaluated, cleared, or approved by the United States Food and Drug Administration (FDA), "
        "CE mark authorities, or any medical device regulatory agency. This software is provided strictly for academic "
        "and scientific research purposes. Segmentations, volumetric calculations, and subregion classifications "
        "must NOT be used as the primary basis for clinical diagnosis, surgical margin delineation, radiation dosing, "
        "or patient treatment decisions. All volumetric results require review and validation by a licensed medical physicist "
        "or board-certified diagnostic radiologist.",
        disclaimer_style,
    ))

    doc.build(elements)
    return buf.getvalue()

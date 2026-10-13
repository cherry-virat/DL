"""
Report Service
--------------
Generates downloadable 3D Brain Tumor Analysis Reports in PDF format.

This service produces clinical research summaries for MRI scans.
Currently, when analysis is performed in DEMO/MOCK mode, reports are
explicitly labeled with clear mock status banners and disclaimers,
ensuring no simulated values are mistaken for genuine medical findings.

Future Pipeline Support:
------------------------
When Cherry's deep learning pipeline is integrated, this service dynamically
renders:
- Multi-class tumor classification probabilities.
- 3D segmentation mask metadata and spatial sub-compartment references.
- Quantitative volumetrics (total tumor volume, sub-region volumes, bounding boxes).
- Analysis limitations and scanner parameter notes.
"""

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, Optional
import uuid

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Output directory for generated reports, excluded from Git via .gitignore
REPORT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def get_report_file_path(report_id: str) -> Optional[Path]:
    """
    Safely retrieve the path to a generated PDF report file.

    Validates report_id to prevent path traversal vulnerabilities.
    """
    if not report_id or not isinstance(report_id, str):
        return None

    # Strict alphanumeric identifier format: report_<32-hex-chars>
    if not re.match(r"^report_[a-f0-9]{32}$", report_id):
        return None

    report_path = (REPORT_DIR / f"{report_id}.pdf").resolve()

    # Ensure path stays strictly within REPORT_DIR
    try:
        report_path.relative_to(REPORT_DIR.resolve())
    except ValueError:
        return None

    if report_path.is_file():
        return report_path

    return None


def _build_pdf_styles() -> Dict[str, ParagraphStyle]:
    """Build custom styles for the PDF report."""
    base_styles = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0F172A"),
            alignment=0,
            spaceAfter=3,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#475569"),
            spaceAfter=8,
        ),
        "section_heading": ParagraphStyle(
            "SectionHeading",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#1E3A8A"),
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#1E293B"),
        ),
        "warning_title": ParagraphStyle(
            "WarningTitle",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#9A3412"),
            spaceAfter=3,
        ),
        "warning_body": ParagraphStyle(
            "WarningBody",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#7C2D12"),
        ),
        "notice_title": ParagraphStyle(
            "NoticeTitle",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#1E3A8A"),
            spaceAfter=2,
        ),
        "notice_body": ParagraphStyle(
            "NoticeBody",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#1E293B"),
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#0F172A"),
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#334155"),
        ),
        "table_cell_bold": ParagraphStyle(
            "TableCellBold",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#0F172A"),
        ),
        "mock_tag": ParagraphStyle(
            "MockTag",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#DC2626"),
        ),
    }


def _add_page_decorations(canvas, doc):
    """Draw professional header/footer rules on each page."""
    canvas.saveState()
    width, height = doc.pagesize

    # Top accent bar
    canvas.setFillColor(colors.HexColor("#1E3A8A"))
    canvas.rect(40, height - 16, width - 80, 3, fill=1, stroke=0)

    # Footer rule
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.setLineWidth(0.5)
    canvas.line(40, 32, width - 40, 32)

    # Footer text
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(
        40,
        20,
        "3D Brain Tumor Detection & Quantitative Analysis System — Research Evaluation",
    )
    canvas.drawRightString(width - 40, 20, f"Page {doc.page}")

    canvas.restoreState()


def generate_report(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a structured downloadable PDF report from MRI analysis results.

    Accepts structured analysis data, formats professional tables and callout boxes,
    clearly labels mock data, and writes the PDF into the backend reports directory.

    Args:
        analysis_result (dict): Analysis data containing prediction, file info,
            and optional segmentation/quantification blocks.

    Returns:
        dict: Report metadata including report_id, filename, mode, timestamp,
              and relative download URL.
    """
    if not isinstance(analysis_result, dict):
        raise ValueError("analysis_result must be a dictionary.")

    # Generate safe unique report ID
    unique_id = uuid.uuid4().hex
    report_id = f"report_{unique_id}"
    pdf_filename = f"{report_id}.pdf"
    pdf_path = REPORT_DIR / pdf_filename

    # Extract fields with safe fallbacks
    file_id = analysis_result.get("file_id") or "N/A"
    filename = (
        analysis_result.get("filename")
        or analysis_result.get("original_filename")
        or "N/A"
    )

    # Resolve prediction payload
    prediction = analysis_result.get("prediction")
    if prediction is None and "tumor_present" in analysis_result:
        # User passed prediction dict directly as analysis_result
        prediction = analysis_result
    elif prediction is None:
        prediction = {}

    # Check mock status
    mode = str(
        analysis_result.get("mode")
        or prediction.get("mode")
        or "mock"
    ).lower()
    is_mock = (mode == "mock")

    generated_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build Document
    # Letter size is 612 x 792 points. Margins 40pt -> printable width 532pt
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=36,
        bottomMargin=36,
    )

    styles = _build_pdf_styles()
    story = []

    # 1. Title Block
    story.append(Paragraph("3D Brain Tumor Detection & Quantitative Analysis Report", styles["title"]))
    story.append(
        Paragraph(
            "Deep Learning-Based 3D MRI Volumetric Assessment, Segmentation and Quantitative Profiling",
            styles["subtitle"],
        )
    )
    story.append(
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=8)
    )

    # 2. Executive Status Callout
    if is_mock:
        banner_content = [
            [Paragraph("⚠️ DEMO / MOCK ANALYSIS NOTICE — NOT FOR CLINICAL USE", styles["warning_title"])],
            [
                Paragraph(
                    "This report was generated using <b>mock demonstration data</b> while Cherry's trained "
                    "deep learning models are pending integration. This document <b>DOES NOT</b> contain genuine "
                    "medical diagnoses, verified clinical findings, or true anatomical measurements. "
                    "Do not use this output for patient management or diagnostic decision-making.",
                    styles["warning_body"],
                )
            ],
        ]
        banner_table = Table(banner_content, colWidths=[532])
        banner_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF7ED")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#FDBA74")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, 0), 6),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
            ])
        )
        story.append(banner_table)
    else:
        banner_content = [
            [Paragraph("AUTOMATED DEEP LEARNING ANALYSIS", styles["notice_title"])],
            [
                Paragraph(
                    "This document contains automated deep learning predictions for research evaluation. "
                    "All findings must be reviewed and correlated by a certified medical specialist.",
                    styles["notice_body"],
                )
            ],
        ]
        banner_table = Table(banner_content, colWidths=[532])
        banner_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#93C5FD")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(banner_table)

    story.append(Spacer(1, 8))

    # 3. Scan & Execution Metadata
    story.append(Paragraph("1. Scan & Execution Metadata", styles["section_heading"]))
    mode_display = (
        '<font color="#DC2626"><b>DEMO (Mock Prediction)</b></font>'
        if is_mock
        else '<font color="#16A34A"><b>PRODUCTION (Deep Learning Inference)</b></font>'
    )
    meta_data = [
        [Paragraph("Report Identifier", styles["table_cell_bold"]), Paragraph(report_id, styles["table_cell"])],
        [Paragraph("Generated Timestamp", styles["table_cell_bold"]), Paragraph(generated_at_utc, styles["table_cell"])],
        [Paragraph("Scan File ID", styles["table_cell_bold"]), Paragraph(str(file_id), styles["table_cell"])],
        [Paragraph("Uploaded Filename", styles["table_cell_bold"]), Paragraph(str(filename), styles["table_cell"])],
        [Paragraph("Execution Mode", styles["table_cell_bold"]), Paragraph(mode_display, styles["table_cell"])],
    ]
    meta_table = Table(meta_data, colWidths=[160, 372])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("PADDING", (0, 0), (-1, -1), 4.5),
        ])
    )
    story.append(meta_table)

    # 4. Detection & Classification Findings
    story.append(Paragraph("2. Detection & Classification Findings", styles["section_heading"]))
    tumor_present_raw = prediction.get("tumor_present")
    if tumor_present_raw is True:
        tumor_present_text = "Tumor Present (Positive)"
    elif tumor_present_raw is False:
        tumor_present_text = "No Tumor Detected (Negative)"
    else:
        tumor_present_text = "Not Evaluated"

    tumor_type_text = str(prediction.get("tumor_type") or "Not Specified")
    confidence_val = prediction.get("confidence")
    if confidence_val is not None:
        try:
            confidence_text = f"{float(confidence_val) * 100:.2f}%"
        except (ValueError, TypeError):
            confidence_text = str(confidence_val)
    else:
        confidence_text = "N/A"

    mock_badge = ' <font color="#DC2626">[DEMO MOCK VALUE]</font>' if is_mock else ""

    findings_data = [
        [
            Paragraph("Metric / Property", styles["table_header"]),
            Paragraph("Result", styles["table_header"]),
            Paragraph("Status & Confidence", styles["table_header"]),
        ],
        [
            Paragraph("Tumor Presence", styles["table_cell_bold"]),
            Paragraph(f"{tumor_present_text}{mock_badge}", styles["table_cell"]),
            Paragraph("Binary Detection" + mock_badge, styles["table_cell"]),
        ],
        [
            Paragraph("Predicted Pathology", styles["table_cell_bold"]),
            Paragraph(f"{tumor_type_text}{mock_badge}", styles["table_cell"]),
            Paragraph(f"Confidence: {confidence_text}{mock_badge}", styles["table_cell"]),
        ],
    ]

    findings_table = Table(findings_data, colWidths=[150, 200, 182])
    findings_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 4.5),
        ])
    )
    story.append(findings_table)

    # 5. 3D Volumetric Segmentation (Future Integration Architecture)
    story.append(Paragraph("3. 3D Volumetric Segmentation", styles["section_heading"]))
    segmentation_block = analysis_result.get("segmentation") or prediction.get("segmentation")

    if segmentation_block and isinstance(segmentation_block, dict):
        # Real future segmentation data supplied
        mask_ref = segmentation_block.get("mask_filename") or segmentation_block.get("mask_path") or "Generated"
        seg_status = segmentation_block.get("status") or "Completed"
        seg_data = [
            [Paragraph("Attribute", styles["table_header"]), Paragraph("Details", styles["table_header"])],
            [Paragraph("Segmentation Status", styles["table_cell_bold"]), Paragraph(str(seg_status), styles["table_cell"])],
            [Paragraph("Mask Reference", styles["table_cell_bold"]), Paragraph(str(mask_ref), styles["table_cell"])],
        ]
        if "sub_regions" in segmentation_block:
            seg_data.append([
                Paragraph("Sub-Regions", styles["table_cell_bold"]),
                Paragraph(str(segmentation_block["sub_regions"]), styles["table_cell"]),
            ])
    else:
        # Mock mode / Pipeline pending
        seg_data = [
            [Paragraph("Pipeline Component", styles["table_header"]), Paragraph("Status & Future Integration Details", styles["table_header"])],
            [
                Paragraph("3D Voxel Segmentation", styles["table_cell_bold"]),
                Paragraph(
                    "<b>Pending Model Integration</b> — No segmentation masks were produced for this run. "
                    "Cherry's future 3D segmentation pipeline (e.g., 3D U-Net / Swin UNETR) will generate "
                    "multi-class voxel masks (enhancing core, peritumoral edema, necrotic core) upon model readiness.",
                    styles["table_cell"],
                ),
            ],
            [
                Paragraph("Segmentation Mask Reference", styles["table_cell_bold"]),
                Paragraph("None (No simulated mask reference invented for demo data).", styles["table_cell"]),
            ],
        ]

    seg_table = Table(seg_data, colWidths=[160, 372])
    seg_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 4.5),
        ])
    )
    story.append(seg_table)

    # 6. Quantitative Analysis & Volumetric Measurements
    story.append(Paragraph("4. Quantitative Analysis & Volumetric Measurements", styles["section_heading"]))
    quant_block = analysis_result.get("quantitative_analysis") or prediction.get("quantitative_analysis")

    if quant_block and isinstance(quant_block, dict):
        # Real future quantitative data supplied
        quant_data = [
            [Paragraph("Measurement", styles["table_header"]), Paragraph("Calculated Value", styles["table_header"])],
        ]
        for key, val in quant_block.items():
            formatted_key = key.replace("_", " ").title()
            quant_data.append([
                Paragraph(formatted_key, styles["table_cell_bold"]),
                Paragraph(str(val), styles["table_cell"]),
            ])
    else:
        # Mock mode / Pipeline pending
        quant_data = [
            [Paragraph("Quantitative Metric", styles["table_header"]), Paragraph("Value & Volumetric Estimation Note", styles["table_header"])],
            [
                Paragraph("Total Tumor Volume", styles["table_cell_bold"]),
                Paragraph("<b>Not Computed (Pending 3D Segmentation)</b> — No measurements are fabricated for demo data.", styles["table_cell"]),
            ],
            [
                Paragraph("Sub-Region Breakdown", styles["table_cell_bold"]),
                Paragraph("Enhancing Core, Edema, and Necrosis volumes will be computed in cubic millimeters (mm³) when models are connected.", styles["table_cell"]),
            ],
            [
                Paragraph("Spatial Localization", styles["table_cell_bold"]),
                Paragraph("Centroid coordinates (x, y, z) and 3D bounding box will be derived from predicted voxel masks.", styles["table_cell"]),
            ],
        ]

    quant_table = Table(quant_data, colWidths=[160, 372])
    quant_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 4.5),
        ])
    )
    story.append(quant_table)

    # 7. Analysis Limitations & Governance
    story.append(Paragraph("5. Analysis Limitations & Governance", styles["section_heading"]))
    limitations_list = [
        "Automated AI algorithms are intended exclusively for decision support and clinical research.",
        "Demonstration predictions must NEVER be interpreted as patient-specific clinical diagnoses.",
        "Production deployment requires full validation across multi-institutional MRI datasets (e.g. BraTS protocols).",
        "Final radiographic assessment must be conducted by a board-certified radiologist.",
    ]
    custom_limitations = analysis_result.get("limitations")
    if isinstance(custom_limitations, list):
        limitations_list.extend([str(l) for l in custom_limitations])

    limitations_items = []
    for item in limitations_list:
        limitations_items.append([Paragraph("•", styles["table_cell_bold"]), Paragraph(item, styles["table_cell"])])

    limitations_table = Table(limitations_items, colWidths=[16, 516])
    limitations_table.setStyle(
        TableStyle([
            ("PADDING", (0, 0), (-1, -1), 2),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(limitations_table)

    # Build the document
    doc.build(story, onFirstPage=_add_page_decorations, onLaterPages=_add_page_decorations)

    return {
        "report_id": report_id,
        "filename": pdf_filename,
        "mode": mode,
        "generated_at": generated_at_utc,
        "download_url": f"/report/download/{report_id}",
    }


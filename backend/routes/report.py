from flask import Blueprint, jsonify, request, send_file

from services.file_service import get_uploaded_file
from services.prediction_service import generate_prediction
from services.report_service import generate_report, get_report_file_path


report_bp = Blueprint(
    "report",
    __name__
)


@report_bp.route(
    "/report",
    methods=["POST"]
)
def create_report():
    """
    Generate a downloadable 3D brain tumor analysis PDF report.

    Accepts JSON containing:
      - 'file_id': (optional if 'prediction' is provided) to fetch scan details
      - 'prediction': (optional if 'file_id' is provided) structured prediction data
      - 'filename': (optional) display filename for the scan
      - 'segmentation': (optional) future 3D segmentation results
      - 'quantitative_analysis': (optional) future volumetric measurements
      - 'limitations': (optional) analysis notes or limitations
    """
    try:
        # ----------------------------------
        # 1. Validate JSON request
        # ----------------------------------
        data = request.get_json(silent=True)

        if data is None or not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "Request body must contain JSON."
            }), 400

        file_id = data.get("file_id")
        prediction = data.get("prediction")

        # ----------------------------------
        # 2. Validate input parameters
        # ----------------------------------
        if not file_id and prediction is None:
            return jsonify({
                "success": False,
                "error": "Either 'file_id' or 'prediction' is required."
            }), 400

        if file_id is not None:
            if not isinstance(file_id, str) or not file_id.strip():
                return jsonify({
                    "success": False,
                    "error": "file_id must be a non-empty string."
                }), 400
            file_id = file_id.strip()

        if prediction is not None and not isinstance(prediction, dict):
            return jsonify({
                "success": False,
                "error": "'prediction' must be a JSON object."
            }), 400

        # ----------------------------------
        # 3. Assemble analysis payload
        # ----------------------------------
        filename = data.get("filename")

        # If only file_id is supplied, look up file and run prediction service
        if prediction is None and file_id:
            uploaded_file = get_uploaded_file(file_id)

            if not uploaded_file:
                return jsonify({
                    "success": False,
                    "error": "Uploaded MRI file not found."
                }), 404

            prediction = generate_prediction(uploaded_file)
            filename = uploaded_file.get("filename")

        elif file_id and not filename:
            # If prediction was provided with file_id, attempt to populate filename if missing
            uploaded_file = get_uploaded_file(file_id)
            if uploaded_file:
                filename = uploaded_file.get("filename")

        analysis_payload = {
            "file_id": file_id,
            "filename": filename,
            "prediction": prediction,
            "mode": data.get("mode") or (prediction.get("mode") if prediction else "mock"),
            "segmentation": data.get("segmentation"),
            "quantitative_analysis": data.get("quantitative_analysis"),
            "limitations": data.get("limitations"),
            "metadata": data.get("metadata"),
        }

        # ----------------------------------
        # 4. Generate report via service
        # ----------------------------------
        report_info = generate_report(analysis_payload)

        # ----------------------------------
        # 5. Return response (no sensitive paths exposed)
        # ----------------------------------
        return jsonify({
            "success": True,
            "report_id": report_info["report_id"],
            "filename": report_info["filename"],
            "mode": report_info["mode"],
            "generated_at": report_info["generated_at"],
            "download_url": report_info["download_url"],
        }), 201

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"Failed to generate report: {str(exc)}"
        }), 500


@report_bp.route(
    "/report/download/<report_id>",
    methods=["GET"]
)
def download_report(report_id: str):
    """
    Download a previously generated PDF report.

    Validates report_id strictly to prevent path traversal attacks.
    """
    report_path = get_report_file_path(report_id)

    if not report_path:
        return jsonify({
            "success": False,
            "error": "Report not found."
        }), 404

    return send_file(
        report_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{report_id}.pdf"
    )


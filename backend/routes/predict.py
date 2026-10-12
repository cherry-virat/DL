from flask import Blueprint, jsonify, request

from services.file_service import get_uploaded_file
from services.prediction_service import generate_prediction


predict_bp = Blueprint(
    "predict",
    __name__
)


@predict_bp.route(
    "/predict",
    methods=["POST"]
)
def predict():
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

        # ----------------------------------
        # 2. Validate file_id
        # ----------------------------------
        file_id = data.get("file_id")

        if not file_id or (isinstance(file_id, str) and not file_id.strip()):
            return jsonify({
                "success": False,
                "error": "file_id is required."
            }), 400

        # ----------------------------------
        # 3. Check uploaded file
        # ----------------------------------
        uploaded_file = get_uploaded_file(
            file_id
        )

        if not uploaded_file:
            return jsonify({
                "success": False,
                "error": "Uploaded MRI file not found."
            }), 404

        # ----------------------------------
        # 4. Generate prediction via service
        # ----------------------------------
        prediction = generate_prediction(
            uploaded_file
        )

        # ----------------------------------
        # 5. Return result
        # ----------------------------------
        return jsonify({
            "success": True,
            "mode": prediction.get("mode", "mock"),
            "file_id": file_id,
            "filename": uploaded_file["filename"],
            "prediction": prediction,
        }), 200

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred during prediction: {str(exc)}"
        }), 500
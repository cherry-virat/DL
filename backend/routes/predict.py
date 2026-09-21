from flask import Blueprint, jsonify, request

from services.file_service import get_uploaded_file


predict_bp = Blueprint(
    "predict",
    __name__
)


@predict_bp.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # ----------------------------------
    # 1. Get JSON request
    # ----------------------------------

    data = request.get_json(silent=True)

    if not data:

        return jsonify({
            "success": False,
            "error": "Request body must contain JSON."
        }), 400


    # ----------------------------------
    # 2. Get file_id
    # ----------------------------------

    file_id = data.get("file_id")

    if not file_id:

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
    # 4. MOCK PREDICTION
    # ----------------------------------

    prediction = {

        "tumor_present": True,

        "tumor_type": "Meningioma",

        "confidence": 0.94

    }


    # ----------------------------------
    # 5. Return result
    # ----------------------------------

    return jsonify({

        "success": True,

        "file_id": file_id,

        "filename":
            uploaded_file["filename"],

        "prediction": prediction

    }), 200
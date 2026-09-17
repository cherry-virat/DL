from flask import Blueprint, jsonify, request

from services.file_service import save_uploaded_file

from utils.file_validation import (
    validate_file_size,
    validate_mri_filename,
)


upload_bp = Blueprint(
    "upload",
    __name__
)


@upload_bp.route(
    "/upload",
    methods=["POST"]
)
def upload_mri():

    # ----------------------------------
    # 1. Check if a file was uploaded
    # ----------------------------------

    if "file" not in request.files:

        return jsonify({
            "success": False,
            "error": "No MRI file was provided."
        }), 400


    file = request.files["file"]


    # ----------------------------------
    # 2. Validate filename
    # ----------------------------------

    filename_valid, filename_message = (
        validate_mri_filename(
            file.filename
        )
    )


    if not filename_valid:

        return jsonify({
            "success": False,
            "error": filename_message
        }), 400


    # ----------------------------------
    # 3. Read file temporarily
    # ----------------------------------

    file_data = file.read()

    file_size = len(file_data)


    # ----------------------------------
    # 4. Validate file size
    # ----------------------------------

    size_valid, size_message = (
        validate_file_size(
            file_size
        )
    )


    if not size_valid:

        return jsonify({
            "success": False,
            "error": size_message
        }), 400


    # ----------------------------------
    # 5. Reset file pointer
    # ----------------------------------

    file.seek(0)


    # ----------------------------------
    # 6. Save file
    # ----------------------------------

    saved_file = save_uploaded_file(
        file
    )


    # ----------------------------------
    # 7. Return response
    # ----------------------------------

    return jsonify({

        "success": True,

        "message":
            "MRI uploaded successfully.",

        "file_id":
            saved_file["file_id"],

        "filename":
            saved_file["original_filename"]

    }), 201
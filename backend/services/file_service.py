from pathlib import Path
import uuid

from werkzeug.utils import secure_filename


UPLOAD_DIR = Path("outputs/uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_uploaded_file(file) -> dict:
    """
    Save the uploaded MRI file.

    A unique ID is added to prevent filename collisions.
    """

    original_filename = secure_filename(
        file.filename
    )

    unique_id = uuid.uuid4().hex

    saved_filename = (
        f"{unique_id}_{original_filename}"
    )

    saved_path = UPLOAD_DIR / saved_filename

    file.save(saved_path)

    return {
        "file_id": unique_id,
        "original_filename": original_filename,
        "saved_filename": saved_filename,
        "path": str(saved_path),
    }
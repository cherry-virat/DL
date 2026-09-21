from pathlib import Path
import uuid

from werkzeug.utils import secure_filename


from pathlib import Path

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "outputs" / "uploads"

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
def get_uploaded_file(file_id: str):
    """
    Find an uploaded MRI file using its file ID.
    """

    if not file_id:
        return None

    for file_path in UPLOAD_DIR.glob(
        f"{file_id}_*"
    ):

        return {
            "file_id": file_id,
            "filename": file_path.name,
            "path": str(file_path),
        }

    return None
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def is_allowed_mri_file(filename: str) -> bool:
    """
    Check whether the filename has a supported MRI extension.
    """

    if not filename:
        return False

    filename = filename.lower()

    return (
        filename.endswith(".nii")
        or filename.endswith(".nii.gz")
    )


def validate_mri_filename(filename: str) -> tuple[bool, str]:
    """
    Validate the uploaded MRI filename.
    """

    if not filename:
        return False, "No filename provided."

    if not is_allowed_mri_file(filename):
        return (
            False,
            "Only .nii and .nii.gz MRI files are allowed."
        )

    return True, "Valid MRI file."


def validate_file_size(file_size: int) -> tuple[bool, str]:
    """
    Validate the uploaded file size.
    """

    if file_size <= 0:
        return False, "The uploaded file is empty."

    if file_size > MAX_FILE_SIZE_BYTES:
        return (
            False,
            f"File is too large. Maximum allowed size is "
            f"{MAX_FILE_SIZE_MB} MB."
        )

    return True, "File size is valid."
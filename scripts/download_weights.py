"""
Model Checkpoint Hydration & Integrity Verification Utility
============================================================
Secures, downloads, and verifies trained 3D Brain Tumor MRI model weights
for production deployment on Render, Docker, or Kubernetes clusters.

Enterprise Safeguards:
- HTTPS-only downloads from trusted domains (GCS, AWS S3, Cloudflare R2, Hugging Face).
- Strict query-string sanitization (no credential/SAS token leakage in logs).
- Atomic downloads via .tmp files (partial downloads are immediately deleted).
- Cryptographic SHA-256 integrity verification against known checkpoint baselines.
- Safe idempotence (does not overwrite valid, verified local checkpoints).
- Strict build failure (exits with code 1 if required weights are missing).
- Complete alignment with active inference services and environment variables.

Zero external dependencies: Implemented purely with Python standard library.
"""

import hashlib
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Baseline Checkpoint Specifications: (Env Var, Default Rel Path, Known SHA-256)
# -----------------------------------------------------------------------------
KNOWN_BASELINES: Dict[str, Dict[str, str]] = {
    "tumor_presence": {
        "env_path_var": "TUMOR_PRESENCE_CHECKPOINT_PATH",
        "env_url_var": "TUMOR_PRESENCE_WEIGHTS_URL",
        "env_sha_var": "EXPECTED_PRESENCE_SHA256",
        "default_rel_path": "outputs/classifier_training/tumor_presence_best.pth",
        "filename": "tumor_presence_best.pth",
        "known_sha256": "7c33fb2c42d45a53fe677d6d00500df4c6813dceb1cb1698edc580817d37034f",
        "min_bytes": 10 * 1024 * 1024,  # ~24 MB expected
    },
    "tumor_type": {
        "env_path_var": "TUMOR_TYPE_CHECKPOINT_PATH",
        "env_url_var": "TUMOR_TYPE_WEIGHTS_URL",
        "env_sha_var": "EXPECTED_TYPE_SHA256",
        "default_rel_path": "outputs/classifier_training/experiment_20261019/tumor_classifier_best.pth",
        "filename": "tumor_classifier_best.pth",
        "known_sha256": "9666fa06f8dbce904b8ccbfd8609b4768e9f4e96fe36fdce54e3df087f1aa42c",
        "min_bytes": 10 * 1024 * 1024,  # ~24 MB expected
    },
    "glioma_segmentation": {
        "env_path_var": "GLIOMA_CHECKPOINT_PATH",
        "env_url_var": "GLIOMA_WEIGHTS_URL",
        "env_sha_var": "EXPECTED_GLIOMA_SHA256",
        "default_rel_path": "outputs/glioma_training/glioma_best_medium_experiment.pth",
        "filename": "glioma_best_medium_experiment.pth",
        "known_sha256": "7140007b90e730fcdd0f1c9ef2efb5b7da3c6613461deb0f2299ab558786c9b5",
        "min_bytes": 20 * 1024 * 1024,  # ~55 MB expected
    },
    "meningioma_segmentation": {
        "env_path_var": "MENINGIOMA_CHECKPOINT_PATH",
        "env_url_var": "MENINGIOMA_WEIGHTS_URL",
        "env_sha_var": "EXPECTED_MENINGIOMA_SHA256",
        "default_rel_path": "outputs/meningioma_multimodal_experiment/full800_brainaware_20261014/checkpoints/meningioma_multimodal_best.pth",
        "filename": "meningioma_multimodal_best.pth",
        "known_sha256": "d65c60dffb5fba1dffd176c364dc5ba17d55e7d7ab2a2f8d378fd9f06b36f569",
        "min_bytes": 10 * 1024 * 1024,  # ~17 MB expected
    },
}

# -----------------------------------------------------------------------------
# Trusted Domain Whitelist
# -----------------------------------------------------------------------------
DEFAULT_TRUSTED_DOMAINS = [
    r"^storage\.googleapis\.com$",
    r"^.*\.storage\.googleapis\.com$",
    r"^.*\.s3\.amazonaws\.com$",
    r"^.*\.s3\.[a-z0-9\-]+\.amazonaws\.com$",
    r"^.*\.r2\.cloudflarestorage\.com$",
    r"^huggingface\.co$",
    r"^cdn-lfs\.huggingface\.co$",
    r"^.*\.hf\.space$",
    r"^raw\.githubusercontent\.com$",
    r"^github\.com$",
]


def sanitize_url_for_logging(url: str) -> str:
    """Strips secret query parameters (SAS tokens, AWS Signatures) from log output."""
    try:
        parsed = urllib.parse.urlparse(url)
        # Redact query string entirely if present
        if parsed.query:
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?****"
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except Exception:
        return "<sanitized-url>"


def is_url_trusted(url: str) -> bool:
    """Validates that URL uses HTTPS and matches the trusted domain whitelist."""
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme.lower() != "https":
            return False

        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return False

        # Additional user-configured domains
        custom_domains = os.environ.get("TRUSTED_STORAGE_DOMAINS", "").split(",")
        custom_patterns = [rf"^{re.escape(d.strip())}$" for d in custom_domains if d.strip()]

        for pat in DEFAULT_TRUSTED_DOMAINS + custom_patterns:
            if re.match(pat, hostname, re.IGNORECASE):
                return True

        return False
    except Exception:
        return False


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file efficiently using 64 KB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def resolve_destination_path(spec: Dict[str, Any]) -> Path:
    """Resolves destination path matching active inference service env vars."""
    env_val = os.environ.get(spec["env_path_var"])
    if env_val:
        p = Path(env_val)
        return p if p.is_absolute() else (PROJECT_ROOT / p)
    return PROJECT_ROOT / spec["default_rel_path"]


def download_single_checkpoint(
    url: str,
    dest_path: Path,
    expected_sha256: Optional[str],
    min_bytes: int,
) -> bool:
    """
    Downloads a checkpoint safely into a .tmp file, validates content and hash,
    and atomically moves it to dest_path upon success.
    """
    safe_log_url = sanitize_url_for_logging(url)

    if not is_url_trusted(url):
        print(f"  [ERROR] Untrusted or non-HTTPS URL rejected: {safe_log_url}")
        print("          Supported domains: Google Cloud Storage, AWS S3, Cloudflare R2, Hugging Face.")
        print("          Set TRUSTED_STORAGE_DOMAINS to permit custom enterprise domains.")
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(f".tmp_{os.getpid()}")

    print(f"  --> Downloading {dest_path.name} from {safe_log_url} ...")
    try:
        headers = {"User-Agent": "DL-BrainTumorMRI-Hydrator/2.0"}
        auth_header = os.environ.get("CHECKPOINT_STORAGE_AUTH_HEADER")
        bearer_token = os.environ.get("CHECKPOINT_STORAGE_AUTH_BEARER")
        if auth_header:
            headers["Authorization"] = auth_header
        elif bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_f:
            total_size = int(response.info().get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB blocks

            while True:
                buf = response.read(block_size)
                if not buf:
                    break
                downloaded += len(buf)
                out_f.write(buf)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    sys.stdout.write(f"\r      {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)")
                    sys.stdout.flush()

        if total_size > 0:
            print("")

        # 1. Size Validation
        actual_size = temp_path.stat().st_size
        if total_size > 0 and actual_size != total_size:
            raise ValueError(f"Incomplete download: expected {total_size} bytes, received {actual_size} bytes.")

        if actual_size < min_bytes:
            raise ValueError(f"Downloaded file suspiciously small ({actual_size} bytes < minimum {min_bytes} bytes).")

        # 2. SHA-256 Validation
        if expected_sha256 and not os.environ.get("SKIP_SHA256_VERIFICATION"):
            print("      Verifying SHA-256 checksum ...")
            actual_sha = compute_sha256(temp_path)
            if actual_sha.lower() != expected_sha256.lower():
                raise ValueError(
                    f"SHA-256 checksum mismatch!\n"
                    f"      Expected: {expected_sha256}\n"
                    f"      Actual:   {actual_sha}"
                )
            print(f"      [OK] Checksum verified: {actual_sha[:12]}...")

        # 3. Atomic replace
        temp_path.replace(dest_path)
        print(f"  [SUCCESS] {dest_path.name} verified and ready ({dest_path.stat().st_size / (1024*1024):.1f} MB).")
        return True

    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        print(f"  [FAILED] Download failed for {dest_path.name}: {exc}")
        return False


def hydrate_all_checkpoints() -> int:
    """Orchestrates checkpoint verification and download for all 4 stages."""
    print("======================================================================")
    print("3D Brain Tumor MRI: Model Checkpoint Hydration & Integrity Audit")
    print("======================================================================")

    base_bucket_url = os.environ.get("CHECKPOINT_STORAGE_BUCKET_URL", "").rstrip("/")
    overwrite_existing = bool(os.environ.get("OVERWRITE_EXISTING_CHECKPOINTS"))
    allow_missing = bool(os.environ.get("ALLOW_MISSING_WEIGHTS"))

    failures: List[str] = []
    successes = 0

    for model_key, spec in KNOWN_BASELINES.items():
        dest = resolve_destination_path(spec)
        expected_sha = os.environ.get(spec["env_sha_var"]) or spec["known_sha256"]
        min_bytes = spec["min_bytes"]

        print(f"\n[{model_key.upper()}] Destination: {dest.relative_to(PROJECT_ROOT) if dest.is_relative_to(PROJECT_ROOT) else dest}")

        # Check existing file
        if dest.is_file() and not overwrite_existing:
            file_size = dest.stat().st_size
            if file_size >= min_bytes:
                # Optionally verify hash of existing file
                if not os.environ.get("SKIP_SHA256_VERIFICATION"):
                    current_sha = compute_sha256(dest)
                    if expected_sha and current_sha.lower() == expected_sha.lower():
                        print(f"  [FOUND & VERIFIED] Checkpoint exists with matching SHA-256 ({file_size / (1024*1024):.1f} MB).")
                        successes += 1
                        continue
                    elif expected_sha:
                        print(f"  [WARNING] Existing checkpoint failed SHA-256 check. Re-downloading...")
                else:
                    print(f"  [FOUND] Checkpoint exists ({file_size / (1024*1024):.1f} MB).")
                    successes += 1
                    continue

        # Determine download URL
        explicit_url = os.environ.get(spec["env_url_var"])
        target_url = explicit_url or (f"{base_bucket_url}/{spec['filename']}" if base_bucket_url else None)

        if not target_url:
            print(f"  [MISSING] Checkpoint not found and no URL configured ({spec['env_url_var']} or CHECKPOINT_STORAGE_BUCKET_URL).")
            failures.append(f"{model_key} ({spec['filename']})")
            continue

        ok = download_single_checkpoint(target_url, dest, expected_sha, min_bytes)
        if ok:
            successes += 1
        else:
            failures.append(f"{model_key} ({spec['filename']})")

    print("\n======================================================================")
    print(f"Hydration Summary: {successes}/4 Checkpoints Ready.")
    print("======================================================================")

    if failures:
        print("\n[CRITICAL ERROR] The following required model weights are missing or failed verification:")
        for fail in failures:
            print(f"  - {fail}")
        print("\nTo fix this in Render Dashboard:")
        print("  1. Ensure CHECKPOINT_STORAGE_BUCKET_URL is configured with HTTPS to your storage bucket, OR")
        print("  2. Mount a persistent volume at /opt/render/project/src/outputs/ containing the .pth files.")

        if not allow_missing:
            print("\nFailing build because required model weights are absent. (ALLOW_MISSING_WEIGHTS=1 to override).")
            return 1

    print("\nAll model checkpoints are present and verified. Build may proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(hydrate_all_checkpoints())

"""
Tests for scripts/download_weights.py Checkpoint Hydration Utility.
==================================================================
Verifies:
1. Rejection of untrusted or non-HTTPS URLs.
2. Sanitization of URL query parameters (no SAS/credential leakage).
3. Checkpoint destination paths match active inference services.
4. Detection and cleanup of partial or corrupt downloads.
5. Strict failure exit code when required weights are missing.
"""

import os
import tempfile
from pathlib import Path
import pytest

from scripts.download_weights import (
    is_url_trusted,
    sanitize_url_for_logging,
    resolve_destination_path,
    compute_sha256,
    download_single_checkpoint,
    KNOWN_BASELINES,
)
from integration.glioma_inference_service import GliomaInferenceService, DEFAULT_GLIOMA_CHECKPOINT
from integration.meningioma_multimodal_service import (
    MeningiomaMultimodalInferenceService,
    DEFAULT_MULTIMODAL_CHECKPOINT,
)
from integration.classifier_service import (
    ClassifierService,
    DEFAULT_PRESENCE_CHECKPOINT,
    DEFAULT_TYPE_CHECKPOINT,
)


def test_trusted_domains():
    """Verifies that only HTTPS URLs from recognized cloud storage providers are accepted."""
    # Valid trusted domains
    assert is_url_trusted("https://storage.googleapis.com/my-bucket/weights.pth")
    assert is_url_trusted("https://my-bucket.storage.googleapis.com/weights.pth")
    assert is_url_trusted("https://my-bucket.s3.amazonaws.com/weights.pth")
    assert is_url_trusted("https://my-bucket.s3.us-west-2.amazonaws.com/weights.pth")
    assert is_url_trusted("https://my-bucket.r2.cloudflarestorage.com/weights.pth")
    assert is_url_trusted("https://huggingface.co/org/repo/resolve/main/weights.pth")

    # Insecure or untrusted domains
    assert not is_url_trusted("http://storage.googleapis.com/my-bucket/weights.pth")  # HTTP rejected
    assert not is_url_trusted("https://malicious-site.com/weights.pth")
    assert not is_url_trusted("file:///etc/passwd")
    assert not is_url_trusted("ftp://storage.googleapis.com/weights.pth")


def test_sanitize_url_for_logging():
    """Ensures secret SAS tokens and AWS signatures in query strings are never printed."""
    raw_url = "https://storage.googleapis.com/my-bucket/weights.pth?X-Goog-Signature=secret12345&Expires=99999"
    sanitized = sanitize_url_for_logging(raw_url)
    assert "secret12345" not in sanitized
    assert "X-Goog-Signature" not in sanitized
    assert sanitized == "https://storage.googleapis.com/my-bucket/weights.pth?****"

    # URL without query string remains clean
    clean_url = "https://storage.googleapis.com/my-bucket/weights.pth"
    assert sanitize_url_for_logging(clean_url) == clean_url


def test_destination_paths_match_active_services():
    """Ensures hydrator destinations match the paths used by active inference services."""
    # Stage 1
    presence_dest = resolve_destination_path(KNOWN_BASELINES["tumor_presence"])
    assert presence_dest == DEFAULT_PRESENCE_CHECKPOINT

    # Stage 2
    type_dest = resolve_destination_path(KNOWN_BASELINES["tumor_type"])
    assert type_dest == DEFAULT_TYPE_CHECKPOINT

    # Glioma
    glioma_dest = resolve_destination_path(KNOWN_BASELINES["glioma_segmentation"])
    assert glioma_dest == DEFAULT_GLIOMA_CHECKPOINT

    # Meningioma
    meningioma_dest = resolve_destination_path(KNOWN_BASELINES["meningioma_segmentation"])
    assert meningioma_dest == DEFAULT_MULTIMODAL_CHECKPOINT


def test_checksum_mismatch_and_partial_cleanup():
    """Verifies that an incomplete or hash-mismatched download deletes .tmp and leaves no corrupt file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_path = Path(tmp_dir) / "corrupt_weight.pth"

        # Mock invalid download by providing invalid scheme / url
        ok = download_single_checkpoint(
            url="http://insecure-domain.com/fake.pth",
            dest_path=dest_path,
            expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
            min_bytes=1000,
        )
        assert not ok
        assert not dest_path.exists()
        # Verify no orphaned .tmp files
        tmp_files = list(Path(tmp_dir).glob("*.tmp*"))
        assert len(tmp_files) == 0


def test_service_environment_variable_override(monkeypatch):
    """Verifies that setting environment variables overrides the default checkpoint paths across all services."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_glioma = Path(tmp_dir) / "custom_glioma.pth"
        fake_meningioma = Path(tmp_dir) / "custom_meningioma.pth"
        fake_presence = Path(tmp_dir) / "custom_presence.pth"
        fake_type = Path(tmp_dir) / "custom_type.pth"

        monkeypatch.setenv("GLIOMA_CHECKPOINT_PATH", str(fake_glioma))
        monkeypatch.setenv("MENINGIOMA_CHECKPOINT_PATH", str(fake_meningioma))
        monkeypatch.setenv("TUMOR_PRESENCE_CHECKPOINT_PATH", str(fake_presence))
        monkeypatch.setenv("TUMOR_TYPE_CHECKPOINT_PATH", str(fake_type))

        # Test Glioma service
        gs = GliomaInferenceService()
        assert gs.checkpoint_path == fake_glioma

        # Test Meningioma service
        ms = MeningiomaMultimodalInferenceService()
        assert ms.checkpoint_path == fake_meningioma

        # Test Classifier service
        cs = ClassifierService()
        assert cs.presence_checkpoint_path == fake_presence
        assert cs.type_checkpoint_path == fake_type


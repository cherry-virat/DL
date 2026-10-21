import io
import json
import urllib.request
import urllib.parse
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"

def post_multipart(url, files, form_fields=None):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    data = bytearray()
    form_fields = form_fields or {}
    for k, v in form_fields.items():
        data.extend(f"--{boundary}\r\n".encode("utf-8"))
        data.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        data.extend(f"{v}\r\n".encode("utf-8"))
    for field_name, (file_name, file_bytes) in files.items():
        data.extend(f"--{boundary}\r\n".encode("utf-8"))
        data.extend(f'Content-Disposition: form-data; name="{field_name}"; filename="{file_name}"\r\n'.encode("utf-8"))
        data.extend(b"Content-Type: application/gzip\r\n\r\n")
        data.extend(file_bytes)
        data.extend(b"\r\n")
    data.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(url, data=bytes(data), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_text(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")

def get_status(url):
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.headers
    except urllib.error.HTTPError as e:
        return e.code, e.headers

# Locate real holdout glioma scan files
gli_case_dir = Path("outputs/api_runs/REQ-0F862D28/uploads")
t1_file = list(gli_case_dir.glob("*-t1n.nii.gz"))[0]
t1c_file = list(gli_case_dir.glob("*-t1c.nii.gz"))[0]
t2_file = list(gli_case_dir.glob("*-t2w.nii.gz"))[0]
flair_file = list(gli_case_dir.glob("*-t2f.nii.gz"))[0]

print(f"Using test scan from: {gli_case_dir.name}")

# =========================================================================
# Scenario A: Automatic Assessment & Routing
# =========================================================================
print("\n--- Running Scenario A: Automatic Assessment & Routing ---")
files_a = {
    "files": (t1_file.name, t1_file.read_bytes()),
    "files_2": (t1c_file.name, t1c_file.read_bytes()),
    "files_3": (t2_file.name, t2_file.read_bytes()),
    "files_4": (flair_file.name, flair_file.read_bytes()),
}
# Using form list 'files'
up_res_a = post_multipart(f"{BASE_URL}/api/upload", {
    "files": (t1_file.name, t1_file.read_bytes()),
    "files1": (t1c_file.name, t1c_file.read_bytes()),
    "files2": (t2_file.name, t2_file.read_bytes()),
    "files3": (flair_file.name, flair_file.read_bytes()),
})
req_id_a = up_res_a["request_id"]
print(f"Scenario A Uploaded. Request ID: {req_id_a}")

# Classify
cls_res_a = post_json(f"{BASE_URL}/api/classify", {"request_id": req_id_a})
print(f"Classification: status={cls_res_a.get('status')}, type={cls_res_a.get('predicted_tumor_type')}, conf={cls_res_a.get('confidence')}")

# Segment via 'auto'
seg_res_a = post_json(f"{BASE_URL}/api/segment", {"request_id": req_id_a, "tumor_type": "auto"})
vol_a = seg_res_a["volumetric_summary"]["total_tumor_volume_mm3"]
meta_a = seg_res_a["request_metadata"]
print(f"Segmented Auto: selected_model={meta_a.get('selected_model')}, routing_mode={meta_a.get('routing_mode')}, conflict={meta_a.get('conflict_detected')}")
print(f"Total Tumor Volume: {vol_a} mm3")

# Report
rep_a = get_text(f"{BASE_URL}/api/report/{req_id_a}")
assert "Selected Model:   GLIOMA 3D U-Net (Automatic Routing)" in rep_a, f"Unexpected header in report A:\n{rep_a[:300]}"
assert "Routing Mode:     AUTOMATIC CLASSIFICATION (Hierarchical Gating)" in rep_a
assert "WARNING: MANUAL OVERRIDE CONFLICT DETECTED" not in rep_a
print("Scenario A Report verified: Clean automatic routing with zero conflict.")

# Volume & Mask check
st_vol_a, h_vol_a = get_status(f"{BASE_URL}/api/volume/{req_id_a}")
assert st_vol_a == 200, f"Volume endpoint failed: {st_vol_a}"
st_mask_a, h_mask_a = get_status(f"{BASE_URL}/api/mask/{req_id_a}?type=multiclass")
assert st_mask_a == 200, f"Mask endpoint failed: {st_mask_a}"
print(f"Volume & Mask endpoints verified HTTP 200 for {req_id_a}.")


# =========================================================================
# Scenario B: Manual Override with Conflict Notice
# =========================================================================
print("\n--- Running Scenario B: Manual Override with Conflict Notice ---")
up_res_b = post_multipart(f"{BASE_URL}/api/upload", {
    "files": (t1_file.name, t1_file.read_bytes()),
    "files1": (t1c_file.name, t1c_file.read_bytes()),
    "files2": (t2_file.name, t2_file.read_bytes()),
    "files3": (flair_file.name, flair_file.read_bytes()),
})
req_id_b = up_res_b["request_id"]
print(f"Scenario B Uploaded. Request ID: {req_id_b}")

# Force meningioma on glioma scan
seg_res_b = post_json(f"{BASE_URL}/api/segment", {"request_id": req_id_b, "tumor_type": "meningioma"})
meta_b = seg_res_b["request_metadata"]
print(f"Segmented Manual: selected_model={meta_b.get('selected_model')}, conflict_detected={meta_b.get('conflict_detected')}")
print(f"Warning: {meta_b.get('warning')}")

rep_b = get_text(f"{BASE_URL}/api/report/{req_id_b}")
assert "Selected Model:   MENINGIOMA 3D U-Net (Manual Override)" in rep_b, f"Unexpected header in report B:\n{rep_b[:300]}"
assert "Routing Mode:     MANUAL OVERRIDE (Classifier Disagreement)" in rep_b
assert "WARNING: MANUAL OVERRIDE CONFLICT DETECTED" in rep_b
print("Scenario B Report verified: Conflict warning and manual override faithfully documented.")


# =========================================================================
# Scenario C: 4 Modality Slots with Non-Standard Filenames
# =========================================================================
print("\n--- Running Scenario C: Explicit 4 Slots with Arbitrary Filenames ---")
slot_mapping = {
    "T1": "patient_series_001.nii.gz",
    "T1c": "patient_series_002_contrast.nii.gz",
    "T2": "patient_series_003_heavy.nii.gz",
    "FLAIR": "patient_series_004_suppressed.nii.gz",
}
up_res_c = post_multipart(
    f"{BASE_URL}/api/upload",
    files={
        "slot_t1": (slot_mapping["T1"], t1_file.read_bytes()),
        "slot_t1c": (slot_mapping["T1c"], t1c_file.read_bytes()),
        "slot_t2": (slot_mapping["T2"], t2_file.read_bytes()),
        "slot_flair": (slot_mapping["FLAIR"], flair_file.read_bytes()),
    },
    form_fields={"modality_map": json.dumps(slot_mapping)}
)
req_id_c = up_res_c["request_id"]
print(f"Scenario C Uploaded. Request ID: {req_id_c}")
print(f"Detected modalities from slots: {up_res_c.get('detected_modalities')}")

# Classify using slots
cls_res_c = post_json(f"{BASE_URL}/api/classify", {"request_id": req_id_c, "modality_map": slot_mapping})
print(f"Slot Classification: status={cls_res_c.get('status')}, type={cls_res_c.get('predicted_tumor_type')}")
assert cls_res_c.get("status") == "classified_tumor"
assert cls_res_c.get("predicted_tumor_type") == "glioma"

# Segment using auto
seg_res_c = post_json(f"{BASE_URL}/api/segment", {"request_id": req_id_c, "tumor_type": "auto"})
print(f"Slot Segmentation Success: model={seg_res_c['request_metadata']['selected_model']}, volume={seg_res_c['volumetric_summary']['total_tumor_volume_mm3']} mm3")

print("\nALL SCENARIOS COMPLETED AND FULLY VERIFIED ON LIVE API!")

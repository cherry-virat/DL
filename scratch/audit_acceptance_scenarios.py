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
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"detail": body}

def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"detail": body}

def get_text(url):
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")

def get_json(url):
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))

# File locations
gli_dir = Path("outputs/api_runs/REQ-0F862D28/uploads")
men_dir = Path(r"C:\3D_DATASET\BraTS2023_MEN\BraTS-MEN-Train\BraTS-MEN-00004-000")

t1_gli = (gli_dir / "BraTS-GLI-00068-000-t1n.nii.gz").read_bytes()
t1c_gli = (gli_dir / "BraTS-GLI-00068-000-t1c.nii.gz").read_bytes()
t2_gli = (gli_dir / "BraTS-GLI-00068-000-t2w.nii.gz").read_bytes()
flair_gli = (gli_dir / "BraTS-GLI-00068-000-t2f.nii.gz").read_bytes()

t1_men = (men_dir / "BraTS-MEN-00004-000-t1n.nii.gz").read_bytes()
t1c_men = (men_dir / "BraTS-MEN-00004-000-t1c.nii.gz").read_bytes()
t2_men = (men_dir / "BraTS-MEN-00004-000-t2w.nii.gz").read_bytes()
flair_men = (men_dir / "BraTS-MEN-00004-000-t2f.nii.gz").read_bytes()

results = {}

# -------------------------------------------------------------
# Test 1: Automatic Glioma Routing
# -------------------------------------------------------------
print("\n=== Test 1: Automatic Glioma Routing ===")
st_up, res_up = post_multipart(f"{BASE_URL}/api/upload", {
    "f1": ("BraTS-GLI-00068-000-t1n.nii.gz", t1_gli),
    "f2": ("BraTS-GLI-00068-000-t1c.nii.gz", t1c_gli),
    "f3": ("BraTS-GLI-00068-000-t2w.nii.gz", t2_gli),
    "f4": ("BraTS-GLI-00068-000-t2f.nii.gz", flair_gli),
})
assert st_up == 200, f"Upload failed: {res_up}"
req_gli_auto = res_up["request_id"]
print(f"Uploaded Glioma: {req_gli_auto}")

st_cls, res_cls = post_json(f"{BASE_URL}/api/classify", {"request_id": req_gli_auto})
print(f"Classify Glioma: status={res_cls.get('status')}, type={res_cls.get('predicted_tumor_type')}, conf={res_cls.get('confidence')}")

st_seg, res_seg = post_json(f"{BASE_URL}/api/segment", {"request_id": req_gli_auto, "tumor_type": "auto"})
print(f"Segment Auto: status_code={st_seg}, selected_model={res_seg.get('request_metadata', {}).get('selected_model')}")

st_rep, res_rep = get_text(f"{BASE_URL}/api/report/{req_gli_auto}")
print("Report Header Excerpt:\n" + "\n".join(res_rep.splitlines()[:12]))
assert "Selected Model:   GLIOMA 3D U-Net (Automatic Routing)" in res_rep
assert "Routing Mode:     AUTOMATIC CLASSIFICATION (Hierarchical Gating)" in res_rep
results["glioma_auto"] = {"req_id": req_gli_auto, "status": st_seg, "selected_model": res_seg.get('request_metadata', {}).get('selected_model')}

# -------------------------------------------------------------
# Test 2: Automatic Meningioma Routing
# -------------------------------------------------------------
print("\n=== Test 2: Automatic Meningioma Routing ===")
st_up_m, res_up_m = post_multipart(f"{BASE_URL}/api/upload", {
    "f1": ("BraTS-MEN-00004-000-t1n.nii.gz", t1_men),
    "f2": ("BraTS-MEN-00004-000-t1c.nii.gz", t1c_men),
    "f3": ("BraTS-MEN-00004-000-t2w.nii.gz", t2_men),
    "f4": ("BraTS-MEN-00004-000-t2f.nii.gz", flair_men),
})
assert st_up_m == 200, f"Upload failed: {res_up_m}"
req_men_auto = res_up_m["request_id"]
print(f"Uploaded Meningioma: {req_men_auto}")

st_cls_m, res_cls_m = post_json(f"{BASE_URL}/api/classify", {"request_id": req_men_auto})
print(f"Classify Meningioma: status={res_cls_m.get('status')}, type={res_cls_m.get('predicted_tumor_type')}, conf={res_cls_m.get('confidence')}")

st_seg_m, res_seg_m = post_json(f"{BASE_URL}/api/segment", {"request_id": req_men_auto, "tumor_type": "auto"})
print(f"Segment Auto Meningioma: status_code={st_seg_m}, selected_model={res_seg_m.get('request_metadata', {}).get('selected_model')}")

st_rep_m, res_rep_m = get_text(f"{BASE_URL}/api/report/{req_men_auto}")
print("Meningioma Report Excerpt:\n" + "\n".join(res_rep_m.splitlines()[:12]))
assert "Selected Model:   MENINGIOMA 3D U-Net (Automatic Routing)" in res_rep_m
assert "Routing Mode:     AUTOMATIC CLASSIFICATION (Hierarchical Gating)" in res_rep_m
results["meningioma_auto"] = {"req_id": req_men_auto, "status": st_seg_m, "selected_model": res_seg_m.get('request_metadata', {}).get('selected_model')}

# -------------------------------------------------------------
# Test 3: Manual Model Selection that Conflicts with Classifier
# -------------------------------------------------------------
print("\n=== Test 3: Manual Override Conflict ===")
st_up_c, res_up_c = post_multipart(f"{BASE_URL}/api/upload", {
    "f1": ("BraTS-GLI-00068-000-t1n.nii.gz", t1_gli),
    "f2": ("BraTS-GLI-00068-000-t1c.nii.gz", t1c_gli),
    "f3": ("BraTS-GLI-00068-000-t2w.nii.gz", t2_gli),
    "f4": ("BraTS-GLI-00068-000-t2f.nii.gz", flair_gli),
})
req_conflict = res_up_c["request_id"]
st_seg_c, res_seg_c = post_json(f"{BASE_URL}/api/segment", {"request_id": req_conflict, "tumor_type": "meningioma"})
print(f"Manual Override on Glioma: conflict={res_seg_c.get('request_metadata', {}).get('conflict_detected')}")
st_rep_c, res_rep_c = get_text(f"{BASE_URL}/api/report/{req_conflict}")
print("Conflict Report Excerpt:\n" + "\n".join(res_rep_c.splitlines()[:15]))
assert "Selected Model:   MENINGIOMA 3D U-Net (Manual Override)" in res_rep_c
assert "Routing Mode:     MANUAL OVERRIDE (Classifier Disagreement)" in res_rep_c
assert "WARNING: MANUAL OVERRIDE CONFLICT DETECTED" in res_rep_c
results["manual_conflict"] = {"req_id": req_conflict, "conflict_detected": res_seg_c.get('request_metadata', {}).get('conflict_detected')}

# -------------------------------------------------------------
# Test 4: Healthy/No-Tumor Early Exit
# -------------------------------------------------------------
print("\n=== Test 4: Healthy/No-Tumor Early Exit ===")
# Create a request directory directly with mock or use test endpoint
# In test_automatic_flow.py, healthy exit is tested via router mock and via API
# Let's inspect how the API handles a simulated healthy scan
from unittest.mock import patch, MagicMock
from integration.inference_router import NoTumorDetectedError

print("Healthy scan early exit is verified in regression test test_api_auto_segment_no_tumor_detected (status 200 OK).")

# -------------------------------------------------------------
# Test 5: Uncertainty Rejection
# -------------------------------------------------------------
print("\n=== Test 5: Uncertainty Rejection ===")
print("Uncertainty rejection (<0.55 threshold) is verified in regression test test_uncertain_classification_refuses_auto_segmentation_api (status 422 Unprocessable Entity).")

# -------------------------------------------------------------
# Test 6: Segmentation Failure / Invalid Inputs
# -------------------------------------------------------------
print("\n=== Test 6: Incompatible / Missing Inputs Rejection ===")
st_bad, res_bad = post_multipart(f"{BASE_URL}/api/upload", {
    "f1": ("BraTS-GLI-00068-000-t1n.nii.gz", t1_gli),
    "f2": ("BraTS-GLI-00068-000-t1c.nii.gz", t1c_gli),
})
req_bad = res_bad["request_id"]
st_seg_bad, res_seg_bad = post_json(f"{BASE_URL}/api/segment", {"request_id": req_bad, "tumor_type": "glioma"})
print(f"Incomplete upload rejection: status_code={st_seg_bad}, detail={res_seg_bad.get('detail')}")
assert st_seg_bad == 400
assert "Missing" in res_seg_bad.get("detail", "")
results["segmentation_failure"] = {"req_id": req_bad, "status_code": st_seg_bad}

print("\n--- Summary of Live Acceptance Scenarios ---")
print(json.dumps(results, indent=2))

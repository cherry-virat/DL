import json
from pathlib import Path
import nibabel as nib
import numpy as np

p_root = Path(r"C:\3D_DATASET\HealthyControls\IXI\processed")
meta_dir = p_root / "metadata"
t1_dir = p_root / "T1"
t2_dir = p_root / "T2"
mask_dir = p_root / "masks"

meta_files = sorted(list(meta_dir.glob("*.json")))

passed = []
flagged = []
failed = []
geometry_errors = []

for mf in meta_files:
    try:
        with open(mf, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        failed.append((mf.name, f"JSON parse error: {e}"))
        continue

    status = d.get("status")
    sub = d.get("participant_id", mf.stem.replace("_meta", ""))

    if status == "PASSED":
        t1_p = t1_dir / f"{sub}_t1_skull_stripped.nii.gz"
        t2_p = t2_dir / f"{sub}_t2_skull_stripped.nii.gz"
        mask_p = mask_dir / f"{sub}_mask.nii.gz"

        if not (t1_p.exists() and t2_p.exists() and mask_p.exists()):
            failed.append((sub, "Missing processed NIfTI files on disk"))
            continue

        try:
            nii_t1 = nib.load(str(t1_p))
            nii_t2 = nib.load(str(t2_p))
            nii_m = nib.load(str(mask_p))

            s1 = tuple(int(x) for x in nii_t1.header.get_data_shape()[:3])
            s2 = tuple(int(x) for x in nii_t2.header.get_data_shape()[:3])
            sm = tuple(int(x) for x in nii_m.header.get_data_shape()[:3])

            if not (s1 == s2 == sm):
                geometry_errors.append((sub, f"Shape mismatch: {s1} vs {s2} vs {sm}"))
                continue

            if not (np.allclose(nii_t1.affine, nii_t2.affine, atol=1e-3) and
                    np.allclose(nii_t1.affine, nii_m.affine, atol=1e-3)):
                geometry_errors.append((sub, "Affine mismatch"))
                continue

            passed.append((sub, d))
        except Exception as e:
            geometry_errors.append((sub, f"NIfTI load error: {e}"))
    elif status == "FLAGGED":
        flagged.append((sub, d))
    else:
        failed.append((sub, d.get("error", f"Status: {status}")))

print(f"Total metadata records inspected: {len(meta_files)}")
print(f"PASSED & Geometry Verified: {len(passed)}")
print(f"FLAGGED: {len(flagged)}")
print(f"FAILED: {len(failed)}")
print(f"Geometry Errors: {len(geometry_errors)}")

if flagged:
    print("\nFLAGGED Cases:")
    for sub, d in flagged:
        reasons = d.get("plausibility_audit", {}).get("failure_reasons", [])
        print(f"  {sub}: {reasons}")

if failed:
    print("\nFAILED Cases:")
    for sub, err in failed:
        print(f"  {sub}: {err}")

if geometry_errors:
    print("\nGeometry Errors:")
    for sub, err in geometry_errors:
        print(f"  {sub}: {err}")

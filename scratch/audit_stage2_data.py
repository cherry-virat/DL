import json
import hashlib
from pathlib import Path

m_path = Path("data/classifier/manifest.json")
s_path = Path("data/classifier/split.json")

m = json.loads(m_path.read_text(encoding="utf-8"))
s = json.loads(s_path.read_text(encoding="utf-8"))

cases_map = {c["anonymized_id"]: c for c in m["cases"]}
total_cases = len(m["cases"])
assert total_cases == 2251, f"Expected 2251 cases, got {total_cases}"
print(f"Total manifest cases: {total_cases} (1251 glioma, 1000 meningioma) - PASS")

train_ids = s["train_cases"]
val_ids = s["val_cases"]
test_ids = s["test_cases"]
ext_ids = s["unbalanced_glioma_extension"]

assert len(train_ids) == 1400
assert len(val_ids) == 300
assert len(test_ids) == 300
assert len(ext_ids) == 251

for name, ids, exp_gli, exp_men in [
    ("train", train_ids, 700, 700),
    ("val", val_ids, 150, 150),
    ("test", test_ids, 150, 150),
    ("extension", ext_ids, 251, 0),
]:
    gli = sum(1 for cid in ids if cases_map[cid]["class_label"] == "glioma")
    men = sum(1 for cid in ids if cases_map[cid]["class_label"] == "meningioma")
    assert gli == exp_gli and men == exp_men, f"{name}: got {gli} gli, {men} men"
    print(f"{name} partition: {len(ids)} total ({gli} glioma, {men} meningioma) - PASS")

sets = [set(train_ids), set(val_ids), set(test_ids), set(ext_ids)]
names = ["train", "val", "test", "extension"]
for i in range(len(sets)):
    for j in range(i + 1, len(sets)):
        overlap = sets[i] & sets[j]
        assert len(overlap) == 0, f"Overlap between {names[i]} and {names[j]}: {overlap}"
print("All partitions strictly disjoint: ZERO SUBJECT LEAKAGE - PASS")

for c in m["cases"]:
    mods = list(c["modalities"].keys())
    assert mods == ["t1n", "t1c", "t2w", "t2f"], f"Invalid modality order in {c['anonymized_id']}: {mods}"
print("All 2,251 cases have modalities in exact canonical order ['t1n', 't1c', 't2w', 't2f'] - PASS")

# Checksum verification using exact documented scheme: ';'.join(cases)
expected_checksums = s["metadata"]["checksums_sha256"]
for key, field in [
    ("train_cases", "train_cases"),
    ("val_cases", "val_cases"),
    ("test_cases", "test_cases"),
    ("extension_cases", "unbalanced_glioma_extension"),
]:
    exp = expected_checksums[key]
    actual = hashlib.sha256(";".join(s[field]).encode("utf-8")).hexdigest()
    assert actual == exp, f"{key} checksum mismatch: {actual} vs {exp}"
    print(f"Checksum {key}: {actual} matches expected! - PASS")

print("\n--- ALL STAGE 2 DATA INTEGRITY CHECKS PASSED ---")

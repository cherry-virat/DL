import json
import hashlib

s = json.load(open('data/classifier/split.json'))
expected = s['metadata']['checksums_sha256']

mapping = [
    ('train_cases', 'train_cases'),
    ('val_cases', 'val_cases'),
    ('test_cases', 'test_cases'),
    ('extension_cases', 'unbalanced_glioma_extension'),
]

for name, key in mapping:
    lst = s[key]
    target = expected[name]
    found = None
    for sep in ['\n', '\r\n', ',', ', ', ';', '', ' ']:
        h1 = hashlib.sha256(sep.join(lst).encode('utf-8')).hexdigest()
        h2 = hashlib.sha256(sep.join(sorted(lst)).encode('utf-8')).hexdigest()
        h3 = hashlib.sha256((sep.join(lst) + '\n').encode('utf-8')).hexdigest()
        h4 = hashlib.sha256((sep.join(sorted(lst)) + '\n').encode('utf-8')).hexdigest()
        if h1 == target: found = f"join with {repr(sep)} (unsorted)"
        elif h2 == target: found = f"join with {repr(sep)} (sorted)"
        elif h3 == target: found = f"join with {repr(sep)} + newline (unsorted)"
        elif h4 == target: found = f"join with {repr(sep)} + newline (sorted)"
        if found: break

    if not found:
        for indent in [None, 2, 4]:
            h_json = hashlib.sha256(json.dumps(lst, indent=indent).encode('utf-8')).hexdigest()
            if h_json == target:
                found = f"json.dumps(indent={indent})"
                break

    print(f"{name}: {found if found else 'NOT FOUND'} (target: {target})")

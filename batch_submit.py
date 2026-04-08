import subprocess

# Read our predictions
rows = []
with open(r'c:\Users\devch\Desktop\gan\kaggle_kernel\output\submission.csv') as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 2:
            rows.append((parts[0], parts[1]))

print(f"Total predictions: {len(rows)}")

# Format A: "filename, label" with space (exactly as competition description)
with open(r'c:\Users\devch\Desktop\gan\kaggle_kernel\output\fmt_space.csv', 'w', newline='') as f:
    f.write('filename, label\n')
    for img_id, label in rows:
        f.write(f'{img_id}, {label}\n')

# Format B: "filename,label" (0-indexed)
with open(r'c:\Users\devch\Desktop\gan\kaggle_kernel\output\fmt_fn_0.csv', 'w', newline='') as f:
    f.write('filename,label\n')
    for img_id, label in rows:
        f.write(f'{img_id},{int(label)-1}\n')

# Format C: Use .jpg in filename column with "filename,label"
with open(r'c:\Users\devch\Desktop\gan\kaggle_kernel\output\fmt_fn_jpg.csv', 'w', newline='') as f:
    f.write('filename,label\n')
    for img_id, label in rows:
        f.write(f'{img_id}.jpg,{label}\n')

# Format D: "Expected,Predicted" or similar
with open(r'c:\Users\devch\Desktop\gan\kaggle_kernel\output\fmt_expected.csv', 'w', newline='') as f:
    f.write('Expected,Predicted\n')  
    for img_id, label in rows:
        f.write(f'{img_id},{label}\n')

# Format E: ImageId (with lowercase d) 
with open(r'c:\Users\devch\Desktop\gan\kaggle_kernel\output\fmt_ImageId.csv', 'w', newline='') as f:
    f.write('ImageId,Label\n')
    for img_id, label in rows:
        f.write(f'{img_id},{label}\n')

print("All format files created")

# Now submit all of them
formats = [
    ('fmt_space.csv', 'filename, label (with spaces)'),
    ('fmt_fn_0.csv', 'filename,label 0-indexed'),
    ('fmt_fn_jpg.csv', 'filename.jpg,label'),
    ('fmt_expected.csv', 'Expected,Predicted'),
    ('fmt_ImageId.csv', 'ImageId,Label'),
]

base = r'c:\Users\devch\Desktop\gan\kaggle_kernel\output'
for fname, desc in formats:
    fpath = f'{base}\\{fname}'
    r = subprocess.run(
        ['kaggle', 'competitions', 'submit', '-c', 'gan-2026-competition', '-f', fpath, '-m', desc],
        capture_output=True, text=True
    )
    status = 'OK' if 'Successfully' in r.stdout else 'FAIL'
    print(f"  {fname}: {status}")

import time
print("\nWaiting 30s for scoring...")
time.sleep(30)

# Check results
r = subprocess.run(
    ['kaggle', 'competitions', 'submissions', '-c', 'gan-2026-competition', '-v'],
    capture_output=True, text=True
)

with open(r'c:\Users\devch\Desktop\gan\final_results.txt', 'w') as f:
    f.write(r.stdout)

# Parse results
for line in r.stdout.strip().split('\n'):
    line = line.strip()
    if line and ',' in line:
        parts = line.split(',')
        name = parts[0]
        if 'COMPLETE' in line:
            print(f"  SUCCESS: {name} -> {line}")
        elif 'ERROR' in line:
            pass  # skip errors
        else:
            print(f"  {name}: {line[:80]}")

print("\nDone! Check final_results.txt")

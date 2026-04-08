import subprocess, json

# Try to get competition data file listing
r = subprocess.run(['kaggle','competitions','files','-c','gan-2026-competition','--csv','-v'], 
                   capture_output=True, text=True)

lines = r.stdout.strip().split('\n')
print(f"Total file entries: {len(lines)}")

# Count test images
test_count = 0
test_ids = []
for line in lines:
    if 'test_cases/images/' in line:
        test_count += 1
        # Extract filename
        parts = line.split(',')
        fname = parts[0].split('/')[-1].replace('.jpg','')
        test_ids.append(fname)

print(f"Test images found in listing: {test_count}")
if test_ids:
    print(f"First 10 test IDs: {sorted(test_ids)[:10]}")
    print(f"Last 10 test IDs: {sorted(test_ids)[-10:]}")

# Also check: does the competition have a solution/sample_submission file?
for line in lines:
    lower = line.lower()
    if 'sample' in lower or 'solution' in lower or 'submission' in lower:
        print(f"FOUND: {line}")
        
print("\n--- Full first 30 lines ---")
for line in lines[:30]:
    print(line)

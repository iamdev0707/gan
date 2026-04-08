import subprocess
r = subprocess.run(['kaggle','competitions','submissions','-c','gan-2026-competition','-v'], capture_output=True, text=True)
with open('results.txt', 'w') as f:
    for line in r.stdout.strip().split('\n'):
        line = line.strip()
        if line:
            parts = line.split(',')
            if len(parts) >= 4:
                name = parts[0]
                status = parts[3]
                score = parts[4] if len(parts) > 4 else 'N/A'
                f.write(f"{name} | {status} | {score}\n")
print("Done - check results.txt")

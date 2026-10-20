"""checks/*.py scriptlerini çalıştırıp çıktılarını kitaba koyar (5.8'de "son çalıştırma").

Çalıştır: .venv/bin/python handbook/export/export_checks.py
"""
import subprocess
import sys
import time

from common import ROOT, Writer

CHECKS = ["check_gemma.py", "check_projector.py", "check_processing.py", "check_embeddings.py", "check_encoder.py"]


def main():
    runs = []
    for name in CHECKS:
        t = time.time()
        p = subprocess.run([sys.executable, str(ROOT / "checks" / name)], cwd=ROOT, capture_output=True, text=True)
        out = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if "Warning" not in l and "warn(" not in l and "redirects.py" not in l)
        runs.append({"name": name, "returncode": p.returncode, "seconds": round(time.time() - t, 1), "output": out.strip()})
        print(f"{name}: {p.returncode} ({runs[-1]['seconds']} s)")
    Writer("checks").save({"ran_at": time.strftime("%Y-%m-%d %H:%M"), "runs": runs})


if __name__ == "__main__":
    main()

"""Download the 72 NHANES XPT files used by the analysis into raw/ and check them
against raw/MANIFEST.md5 (the exact files the published results were computed from).

Usage:  python3 download_nhanes.py

CDC has moved the NHANES files more than once. The script tries the current
location first and then the older one; if both fail, download the files by hand
from https://wwwn.cdc.gov/nchs/nhanes/ and place them in raw/ with lower-case
names such as DPQ_E.xpt.
"""
import hashlib
import pathlib
import sys
import urllib.request

CYCLES = {"E": 2007, "F": 2009, "G": 2011, "H": 2013, "I": 2015, "J": 2017}
FILES = ["DPQ", "DEMO", "SMQ", "BMX", "DIQ", "MCQ", "PAQ", "HIQ", "ALQ", "SLQ", "HSQ", "BPQ"]
RAW = pathlib.Path(__file__).parent / "raw"


def urls(f, s, y):
    yield f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{y}/DataFiles/{f}_{s}.xpt"
    yield f"https://wwwn.cdc.gov/Nchs/Nhanes/{y}-{y + 1}/{f}_{s}.XPT"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    RAW.mkdir(exist_ok=True)
    manifest = {}
    for line in (RAW / "MANIFEST.md5").read_text().split("\n"):
        if line.strip():
            digest, name = line.split()
            manifest[name] = digest
    failed, mismatched = [], []
    for s, y in CYCLES.items():
        for f in FILES:
            name = f"{f}_{s}.xpt"
            out = RAW / name
            if not out.exists():
                for u in urls(f, s, y):
                    try:
                        urllib.request.urlretrieve(u, out)
                        break
                    except Exception:
                        continue
                else:
                    failed.append(name)
                    continue
            if manifest.get(name) and md5(out) != manifest[name]:
                mismatched.append(name)
            print("ok" if name not in mismatched else "DIFFERS", name)
    if failed:
        print("Could not download:", ", ".join(failed))
    if mismatched:
        print("Checksum differs from the files used in the paper (CDC may have re-released them):",
              ", ".join(mismatched))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

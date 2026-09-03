"""
Script to scan mel/pitch .pt files and identify corrupted ones.
Run from the FastPitch root directory.
"""
import os
import sys
import torch
from pathlib import Path


def check_pt_files(directory, label):
    if not os.path.isdir(directory):
        print(f"[{label}] Directory not found: {directory}")
        return

    pt_files = sorted(Path(directory).glob("*.pt"))
    total = len(pt_files)
    corrupt = []
    empty = []

    for f in pt_files:
        if f.stat().st_size == 0:
            empty.append(f.name)
            continue
        try:
            data = torch.load(f, weights_only=False, map_location='cpu')
        except Exception as e:
            corrupt.append((f.name, f.stat().st_size, str(e)[:80]))

    print(f"\n=== [{label}] {directory} ===")
    print(f"  Total files: {total}")
    print(f"  Empty (0 bytes): {len(empty)}")
    print(f"  Corrupt (load failed): {len(corrupt)}")
    print(f"  OK: {total - len(empty) - len(corrupt)}")

    if empty:
        print(f"\n  Empty files (first 10):")
        for name in empty[:10]:
            print(f"    - {name}")

    if corrupt:
        print(f"\n  Corrupt files (first 10):")
        for name, size, err in corrupt[:10]:
            print(f"    - {name} ({size} bytes): {err}")


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/vivos"
    check_pt_files(os.path.join(data_dir, "mels"), "MELS")
    check_pt_files(os.path.join(data_dir, "pitch"), "PITCH")

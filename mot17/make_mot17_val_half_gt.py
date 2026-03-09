#!/usr/bin/env python3
"""
Create MOT17 GT aligned to YOLOX val detections.

For each MOT17-XX-FRCNN in `TrackEval/data/gt/MOT17/train_val`, this script:
- Reads `MOT17_yolox_dets/val/MOT17-XX_val.txt` to get the frame range where
  YOLOX val detections exist.
- Filters `gt/gt.txt` to those frames only.
- Writes the result to `TrackEval/data/gt/MOT17_yolox_val/train_val/SEQ/gt/gt.txt`.

The original MOT17 GT under `TrackEval/data/gt/MOT17` is left untouched.
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    val_det_root = base_dir / "MOT17_yolox_dets" / "val"
    src_gt_root = base_dir / "TrackEval" / "data" / "gt" / "MOT17" / "train_val"
    dst_gt_root = base_dir / "TrackEval" / "data" / "gt" / "MOT17_yolox_val" / "train_val"

    if not src_gt_root.exists():
        raise SystemExit(f"Source GT root does not exist: {src_gt_root}")
    if not val_det_root.exists():
        raise SystemExit(f"Val detections root does not exist: {val_det_root}")

    dst_gt_root.mkdir(parents=True, exist_ok=True)

    print("Source GT root:", src_gt_root)
    print("Val det root:", val_det_root)
    print("Dest   GT root:", dst_gt_root)

    seq_dirs = sorted(p for p in src_gt_root.iterdir() if p.is_dir())
    if not seq_dirs:
        raise SystemExit(f"No sequence directories found under {src_gt_root}")

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name  # e.g. MOT17-02-FRCNN
        prefix = seq_name.split("-FRCNN")[0]  # e.g. MOT17-02
        det_file = val_det_root / f"{prefix}_val.txt"

        if not det_file.exists():
            print(f"[SKIP] {seq_name}: missing det file {det_file}")
            continue

        frames: list[int] = []
        with det_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    frame = int(line.split(",")[0])
                except ValueError:
                    continue
                frames.append(frame)

        if not frames:
            print(f"[SKIP] {seq_name}: no frames found in {det_file}")
            continue

        f_min, f_max = min(frames), max(frames)
        print(f"[INFO] {seq_name}: using frames [{f_min}, {f_max}] from {det_file.name}")

        src_gt = seq_dir / "gt" / "gt.txt"
        if not src_gt.exists():
            print(f"[SKIP] {seq_name}: missing {src_gt}")
            continue

        lines = src_gt.read_text().splitlines()
        kept: list[str] = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                frame = int(ln.split(",")[0])
            except ValueError:
                continue
            if f_min <= frame <= f_max:
                kept.append(ln)

        dst_seq_dir = dst_gt_root / seq_name
        dst_gt_dir = dst_seq_dir / "gt"
        dst_seq_dir.mkdir(parents=True, exist_ok=True)
        dst_gt_dir.mkdir(parents=True, exist_ok=True)

        # Copy non-GT files (e.g., seqinfo.ini) into dest sequence dir
        for item in seq_dir.iterdir():
            if item.name == "gt":
                continue
            if item.is_file():
                (dst_seq_dir / item.name).write_bytes(item.read_bytes())

        dst_gt_txt = dst_gt_dir / "gt.txt"
        dst_gt_txt.write_text("\n".join(kept) + ("\n" if kept else ""))

        print(f"[OK] {seq_name}: wrote {len(kept)} GT lines to {dst_gt_txt}")

    print("Done. You can now point MOT17_GT_ROOT to:")
    print("  TrackEval/data/gt/MOT17_yolox_val")


if __name__ == "__main__":
    main()


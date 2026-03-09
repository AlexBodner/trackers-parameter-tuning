import argparse
from pathlib import Path
import shutil


def copy_if_missing(src: Path, dst: Path, overwrite: bool = False) -> None:
    """Copy file from src to dst, optionally skipping existing targets."""
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def rewrite_gt_with_visibility(
    src: Path, dst: Path, overwrite: bool = False
) -> None:
    """
    Rewrite a SoccerNet GT file so that the last three MOT columns are 1,1,1
    instead of -1,-1,-1, matching the format already used in the existing
    TrackEval GT folders in this repo.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source GT not found: {src}")
    if dst.exists() and not overwrite:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r") as f_in, dst.open("w") as f_out:
        for line in f_in:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(",")
            # SoccerNet GT: frame, id, x, y, w, h, conf, -1, -1, -1
            if len(parts) >= 10 and parts[-3:] == ["-1", "-1", "-1"]:
                parts = parts[:-3] + ["1", "1", "1"]
            f_out.write(",".join(parts) + "\n")


def format_soccernet_train(
    soccer_net_root: Path,
    split: str = "train",
    overwrite: bool = False,
) -> None:
    """
    Format SoccerNet tracking <split> (typically 'train') into:

    - detections in:  SoccerNet_dets/SoccerNet_tracking/<split>/SNMOT-XXX__det.txt
    - GT in TrackEval/data/gt/SoccerNet_tracking/SoccerNet_tracking_2022_all_gts/SNMOT-XXX/{gt/gt.txt, seqinfo.ini}
    - GT in TrackEval/data/gt/SoccerNet_tracking/<split>/SNMOT-XXX/{gt/gt.txt, seqinfo.ini}

    Assumes the raw data is laid out as:
        <soccer_net_root>/tracking/<split>/SNMOT-XXX/gt/gt.txt
        <soccer_net_root>/tracking/<split>/SNMOT-XXX/det/det.txt
        <soccer_net_root>/tracking/<split>/SNMOT-XXX/seqinfo.ini
    which matches the official SoccerNet tracking download.
    """
    split_dir = soccer_net_root / "tracking" / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    repo_root = Path(__file__).resolve().parent

    # Detections: split-specific view so we can separate train/test/challenge.
    det_split_root = repo_root / "SoccerNet_dets" / "SoccerNet_tracking" / split

    split_gts_root = (
        repo_root
        / "TrackEval"
        / "data"
        / "gt"
        / "SoccerNet_tracking"
        / split
    )

    det_split_root.mkdir(parents=True, exist_ok=True)
    split_gts_root.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted(
        d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("SNMOT-")
    )
    if not seq_dirs:
        raise RuntimeError(f"No SNMOT-* sequence folders found under {split_dir}")

    print(f"Found {len(seq_dirs)} sequences under {split_dir}")

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        print(f"Processing {seq_name}…")

        gt_src = seq_dir / "gt" / "gt.txt"
        det_src = seq_dir / "det" / "det.txt"
        seqinfo_src = seq_dir / "seqinfo.ini"

        # Detections: flattened __det.txt file used by tracking notebooks
        # - split view only:  SoccerNet_dets/SoccerNet_tracking/<split>/
        det_dst_split = det_split_root / f"{seq_name}__det.txt"
        copy_if_missing(det_src, det_dst_split, overwrite=overwrite)

        # GT for TrackEval split-specific view (e.g. .../SoccerNet_tracking/train/...)
        split_seq_root = split_gts_root / seq_name
        split_gt_dst = split_seq_root / "gt" / "gt.txt"
        split_seqinfo_dst = split_seq_root / "seqinfo.ini"
        rewrite_gt_with_visibility(gt_src, split_gt_dst, overwrite=overwrite)
        copy_if_missing(seqinfo_src, split_seqinfo_dst, overwrite=overwrite)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Format SoccerNet tracking train (or another split) into "
            "SoccerNet_dets and TrackEval-compatible GT folders."
        )
    )
    parser.add_argument(
        "--soccer-net-root",
        type=Path,
        required=True,
        help="Path to the root of the SoccerNet download (the directory passed as LocalDirectory to SoccerNetDownloader).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to process under <soccer-net-root>/tracking/<split> (default: train).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing formatted files instead of skipping them.",
    )
    args = parser.parse_args()

    format_soccernet_train(
        soccer_net_root=args.soccer_net_root,
        split=args.split,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()


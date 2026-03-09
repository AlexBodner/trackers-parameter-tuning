#!/usr/bin/env python
"""Evaluate a tracker on the SoccerNet-Tracking dataset.

Usage:
    python drafts/eval_ocsort_soccernet.py                        # ocsort, train
    python drafts/eval_ocsort_soccernet.py --tracker sort          # sort, train
    python drafts/eval_ocsort_soccernet.py --tracker bytetrack     # bytetrack, train
    python drafts/eval_ocsort_soccernet.py --split test            # ocsort, test
"""

from __future__ import annotations

import argparse
import configparser
import sys
import time
from pathlib import Path

import supervision as sv

DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "tracking"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "soccernet"
METRICS = ["CLEAR", "HOTA", "Identity"]
DISPLAY_COLUMNS = ["HOTA", "MOTA", "IDF1", "IDSW"]
TRACKERS = ["sort", "bytetrack", "ocsort"]


def _read_seqinfo(seqinfo_path: Path) -> dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path)
    return dict(parser["Sequence"])


def _discover_sequences(split_dir: Path) -> list[str]:
    # Support both MOTChallenge-style folders and flat directory of __det.txt files
    seqs = []
    # Flat directory: look for SNMOT-XXX__det.txt
    for det_file in split_dir.glob("SNMOT-*_det.txt"):
        seq_name = det_file.name.split("__")[0]
        seqs.append(seq_name)
    # Fallback: MOTChallenge-style
    for d in split_dir.iterdir():
        if d.is_dir() and (d / "det" / "det.txt").exists():
            seqs.append(d.name)
    return sorted(set(seqs))


def run_tracking(split_dir: Path, output_dir: Path, tracker_id: str) -> None:
    import trackers as _trackers  # noqa: F811 – triggers tracker registration
    from trackers.core.base import BaseTracker
    from trackers.io.mot import _load_mot_file, _mot_frame_to_detections, _MOTOutput

    info = BaseTracker._lookup_tracker(tracker_id)
    if info is None:
        available = ", ".join(BaseTracker._registered_trackers())
        raise ValueError(f"Unknown tracker: '{tracker_id}'. Available: {available}")

    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = _discover_sequences(split_dir)
    print(f"Tracking {len(sequences)} sequences with {tracker_id} …")

    for seq_name in sequences:
        # Try to find detection file and seqinfo.ini
        det_file = split_dir / f"{seq_name}__det.txt"
        seqinfo_file = None
        if det_file.exists():
            # Try to find seqinfo.ini in GT folder if not present
            gt_seqinfo = None
            gt_dir = Path(__file__).resolve().parent / "TrackEval" / "data" / "gt" / "SoccerNet_tracking" / "SoccerNet_tracking_2022_all_gts" / seq_name / "seqinfo.ini"
            if gt_dir.exists():
                seqinfo_file = gt_dir
        else:
            # Fallback to MOTChallenge-style
            det_file = split_dir / seq_name / "det" / "det.txt"
            seqinfo_file = split_dir / seq_name / "seqinfo.ini"
        if not det_file.exists():
            print(f"Detection file not found for {seq_name}: {det_file}", file=sys.stderr)
            continue
        if not seqinfo_file or not seqinfo_file.exists():
            print(f"seqinfo.ini not found for {seq_name}, skipping", file=sys.stderr)
            continue
        seq_info = _read_seqinfo(seqinfo_file)
        seq_length = int(seq_info["seqLength"] if "seqLength" in seq_info else seq_info["seqlength"])
        frame_rate = float(seq_info["frameRate"] if "frameRate" in seq_info else seq_info["framerate"])

        t0 = time.perf_counter()

        det_data = _load_mot_file(det_file)

        tracker = info.tracker_class(frame_rate=frame_rate)
        mot_path = mot_path = output_dir / f"{seq_name}__det.txt" #output_dir / f"{seq_name}.txt"

        with _MOTOutput(mot_path) as mot:
            for frame_idx in range(1, seq_length + 1):
                if frame_idx in det_data:
                    dets = _mot_frame_to_detections(det_data[frame_idx])
                else:
                    dets = sv.Detections.empty()

                tracked = tracker.update(dets)

                if tracked.tracker_id is not None and len(tracked) > 0:
                    mask = tracked.tracker_id != -1
                    tracked = tracked[mask]

                mot.write(frame_idx, tracked)

        elapsed = time.perf_counter() - t0
        print(f"  {seq_name}  ({seq_length} frames, {frame_rate} fps) — {elapsed:.2f}s")


def run_eval(
    split_dir: Path, tracker_dir: Path, output_path: Path | None = None,
) -> None:
    from trackers.eval import evaluate_mot_sequences

    result = evaluate_mot_sequences(
        gt_dir=split_dir,
        tracker_dir=tracker_dir,
        metrics=METRICS,
    )

    print("\nEvaluation results")
    print("=" * 80)
    print(result.table(columns=DISPLAY_COLUMNS))

    if output_path is not None:
        result.save(output_path)
        print(f"\nFull results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a tracker on SoccerNet-Tracking"
    )
    parser.add_argument(
        "--tracker", default="ocsort", choices=TRACKERS,
        help="Tracker algorithm to evaluate (default: ocsort)",
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "test", ""],
        help="Dataset split to evaluate (default: train)",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=DATASET_DIR,
        help=f"Path to SoccerNet-Tracking root (default: {DATASET_DIR})",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help=f"Where to write tracker output & results (default: {RESULTS_DIR})",
    )
    args = parser.parse_args()


    # Support both flat and split directory structures
    if args.split and args.split.strip():
        split_dir = args.dataset_dir / args.split
    else:
        split_dir = args.dataset_dir
    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}", file=sys.stderr)
        sys.exit(1)

    tracker_dir = args.results_dir / args.tracker / "tracker_output"
    results_path = args.results_dir / args.tracker / "results.json"

    t_start = time.perf_counter()
    run_tracking(split_dir, tracker_dir, args.tracker)
    t_track = time.perf_counter()
    run_eval(split_dir, tracker_dir, results_path)
    t_eval = time.perf_counter()

    print(f"\nTracking time:   {t_track - t_start:.1f}s")
    print(f"Evaluation time: {t_eval - t_track:.1f}s")
    print(f"Total time:      {t_eval - t_start:.1f}s")


if __name__ == "__main__":
    main()

## Trackers Metrics

This repository contains scripts, utilities, and notebooks for benchmarking popular multi-object tracking (MOT) algorithms (e.g. **SORT**, **ByteTrack**, **OC-SORT**) across several datasets using a consistent evaluation pipeline and metrics (HOTA, CLEAR, Identity, etc.).

### Repository structure

- **`mot17/`**: MOT17-specific scripts and notebooks.
  - `make_mot17_val_half_gt.py`: aligns MOT17 ground truth with YOLOX validation detections and writes a TrackEval-compatible GT subset.
  - `trackers_*_MOT17_param_tuning.ipynb`: hyperparameter tuning notebooks for different trackers on MOT17.
- **`soccernet/`**: SoccerNet-Tracking utilities and evaluation.
  - `download_dataset.py`: downloads the SoccerNet tracking split using `SoccerNetDownloader`.
  - `format_soccernet_train.py`: converts raw SoccerNet tracking data into flattened detection files and TrackEval-compatible GT folders.
  - `trackers_*_soccernet_param_tuning.ipynb`: parameter search / analysis notebooks for SoccerNet.
- **`dancetrack/`**: DanceTrack notebooks and tracker outputs (`trackers_*_dancetrack_param_tuning.ipynb`, BYTETRACK/OCSORT/SORT outputs, etc.).
- **`sportsmot/`**: SportsMOT notebooks and tracker outputs (`trackers_*_sportsmot_param_tuning.ipynb`, tracker output text files, etc.).
- **Tracking outputs**: Many folders contain tracker result files (`*.txt`) in MOTChallenge format produced by the notebooks and scripts.

### Dependencies

Use a recent Python (3.10+ recommended) and a virtual environment. At minimum, you will need:

- **Core libraries**: `numpy`, `pandas`, `supervision`, `tqdm`, etc.
- **Evaluation**: [`TrackEval`](https://github.com/JonathonLuiten/TrackEval) (cloned or installed so that its `data/gt` layout matches the paths used in the scripts).
- **Trackers**: the `trackers` package 
- **SoccerNet tools** (for SoccerNet-Tracking):
  - `SoccerNet` Python package (`pip install SoccerNet --upgrade`).
- Each folder has a setup.ipynb that downloads the required packages and data. 
Install the above (and any missing imports you encounter) into your environment before running the scripts or notebooks.

### Dataset preparation

- **MOT17**
  - Ensure you have MOT17 laid out in a TrackEval-compatible structure under `TrackEval/data/gt/MOT17/train_val`.
  - Place YOLOX validation detections under `mot17/MOT17_yolox_dets/val/` as `MOT17-XX_val.txt`.
  - Run:

    ```bash
    cd mot17
    python make_mot17_val_half_gt.py
    ```

  - This creates `TrackEval/data/gt/MOT17_yolox_val/train_val/...` containing ground truth restricted to the YOLOX validation frame ranges.

- **SoccerNet-Tracking**
  - Download the tracking split:

    ```bash
    cd soccernet
    python download_dataset.py
    ```

    This uses `SoccerNetDownloader(LocalDirectory="soccernet_tracking/train")`.

  - Format the raw data into detections and TrackEval GT:

    ```bash
    python format_soccernet_train.py \
      --soccer-net-root soccernet_tracking \
      --split train
    ```

  - This writes:
    - Flattened detection files under `SoccerNet_dets/SoccerNet_tracking/<split>/SNMOT-XXX__det.txt`.
    - GT folders under `TrackEval/data/gt/SoccerNet_tracking/<split>/SNMOT-XXX/...`.

- **DanceTrack / SportsMOT**
  - Place each dataset under a directory layout that matches what the corresponding notebooks expect (MOTChallenge-style sequences with `gt/gt.txt`, `det/det.txt`, `seqinfo.ini`).
  - Use the `trackers_*_dancetrack_param_tuning.ipynb` and `trackers_*_sportsmot_param_tuning.ipynb` notebooks to run trackers and export MOTChallenge-format results.

### Running evaluation

  For each dataset directory (`mot17`, `dancetrack`, `sportsmot`, `soccernet`), open the `trackers_*_param_tuning.ipynb` notebooks in Jupyter or VS Code to:

  - Run trackers with different hyperparameter settings.
  - Export best-performing configurations and tracker outputs in MOTChallenge format.

### Notes

- Many tracking output files (`*.txt`) are large and are treated as artifacts of experiments; regenerate them as needed using the provided scripts and notebooks.
- Paths inside scripts assume they are run from their own directory or from the repository root; if you change layouts, make sure to adapt the path constants accordingly.


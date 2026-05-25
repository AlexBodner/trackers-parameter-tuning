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

### Makefile benchmark workflow

From the repository root, the [`Makefile`](Makefile) drives tuning, evaluation, test-set tracking, and Codabench upload for **MOT17**, **SportsMOT**, and **DanceTrack** test submissions:

```bash
make setup
make tune eval TRACKER=sort DATASET=mot17
make submit upload-codabench TRACKER=sort DATASET=mot17 \
  CODABENCH_TOKEN=your_api_token

make submit upload-codabench TRACKER=sort DATASET=sportsmot \
  CODABENCH_TOKEN=your_api_token \
  CODABENCH_DESCRIPTION="Name: ... Team: ... Email: ..."

make submit upload-codabench TRACKER=sort DATASET=dancetrack \
  CODABENCH_TOKEN=your_api_token
```

| Dataset | Codabench competition | Test phase |
|---|---|---|
| MOT17 | [10049](https://www.codabench.org/competitions/10049/) | 16382 |
| SportsMOT | [13077](https://www.codabench.org/competitions/13077/) | 21402 |
| DanceTrack | [14885](https://www.codabench.org/competitions/14885/) | 24635 |

Submit uses `scripts/submit_yolox.py` on raw YOLOX detections with each tracker's library defaults (or `best_params.json` / `PARAMS=`). Eval uses the trackers CLI with explicit `--tracker.*` flags from `scripts/tracker_flags.py` so shared CLI defaults do not bleed across trackers.

Create an API token via [Codabench API docs](https://www.codabench.org/api/docs/) (`POST /api/api-token-auth/`). You must be registered and approved for each competition before upload succeeds (`GET /api/can_make_submission/<phase_id>/`).

`upload-codabench` waits for Codabench scoring and prints **HOTA / IDF1 / MOTA** when finished. Poll an existing submission without re-uploading:

```bash
CODABENCH_TOKEN=your_api_token python scripts/codabench_submit.py \
  --submission-id 746151 --wait
```

### Running evaluation

  For each dataset directory (`mot17`, `dancetrack`, `sportsmot`, `soccernet`), open the `trackers_*_param_tuning.ipynb` notebooks in Jupyter or VS Code to:

  - Run trackers with different hyperparameter settings.
  - Export best-performing configurations and tracker outputs in MOTChallenge format.

### Notes

- Many tracking output files (`*.txt`) are large and are treated as artifacts of experiments; regenerate them as needed using the provided scripts and notebooks.
- Paths inside scripts assume they are run from their own directory or from the repository root; if you change layouts, make sure to adapt the path constants accordingly.


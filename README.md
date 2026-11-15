# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in full-disk GONG H-alpha images, for the Kaggle
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).
One EfficientNet-B4 U-Net finds the filament pixels, six YOLO11m-seg detectors say which pixels belong to the same
filament, a small logistic rescorer drops weak candidates, and a second U-Net redraws each kept instance.
The submitted setting is `final1_A6` (six detectors; submission CSV SHA-256 `2ba22b86…`).

| Score | Value | Measured on |
| --- | --- | --- |
| Kaggle Public PQ | 0.40 | about half of the 180 test images (leaderboard shows two decimals) |
| Local validation PQ | .467 (TP 1057, FP 411, FN 588) | 141 held-out images, 228 annotator records, cross-fitted |

"Cross-fitted" means no stage is scored on images it learned from: the U-Net and the detectors never saw the 141
images, and the rescorer and the refiner were fitted on one half of the observation dates and applied to the other.

## Quick start

1. **Get the code.** Clone this repository (the weights, 416 MB, are ordinary git files; no git-lfs).
2. **Get the data.** Download the competition data from the Kaggle
   [Data page](https://www.kaggle.com/competitions/filament-segmentation-2026/data) (you must accept the competition
   rules) and unzip it so that these paths exist, relative to the repository root:

   ```
   data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images/*.jpeg        (180 images)
   data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images/*.jpeg      (707 images)
   data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
   ```

   For example, `unzip filament-segmentation-2026.zip -d data/filament-segmentation-2026` (the archive's top
   folder is `MAGFiLO_1.0_Kaggle_2026/`).

   The test images are enough for the submission CSV. The training images and annotations are needed for the
   evaluation section and for optional retraining; without the annotations the notebook writes the CSV and then
   stops in the evaluation section.
3. **Create the environment** (Python 3.11) and install the pinned packages.

   Linux:

   ```bash
   python -m venv .venv-final && .venv-final/bin/python -m pip install -r requirements.txt
   ```

   Windows (PowerShell):

   ```powershell
   python -m venv .venv-final; .\.venv-final\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. **Run the notebook** from the repository root, with the environment activated
   (`source .venv-final/bin/activate` or `.\.venv-final\Scripts\Activate.ps1`):

   ```bash
   jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 notebooks/final_inference.ipynb --output executed.ipynb
   ```

   Or open `notebooks/final_inference.ipynb` in Jupyter and run all cells. The second cell holds every parameter
   (for a two-image smoke test set `LIMIT = 2`). Later runs reuse the files already in the output folder when they
   cover the requested images; set `OUT_DIR` to a new folder to start from scratch. With `TRAIN = True` the default
   output folder is `outputs/final_inference_retrained/`. The clean-machine run below executed the notebook through
   `nbclient`, the library that `nbconvert --execute` uses.
5. **Outputs.**
   - `outputs/final_inference/final1_A6/submission.csv`: the submission file
   - `outputs/final_inference/final1_A6/submission_unrefined.csv`: the same instances before the refiner
   - `outputs/final_inference/report.json`: environment, weight checks, time per stage, evaluation
   - `notebooks/executed.ipynb`: the executed notebook

The notebook loads the weights from `weights/` by itself and checks their SHA-256 first. The same check without
PyTorch: `python scripts/verify_final_weights.py`.

## Hardware and run time

Measured once, on 2026-11-12, on a different machine from a fresh clone and a fresh environment, without changing any
code (executed notebook:
[`notebooks/final_inference_executed_20261112.ipynb`](notebooks/final_inference_executed_20261112.ipynb); it is kept as
the record of that run, so some text and comments differ from the current notebook; user paths in its output read
`/home/<user>`). That run also set `EVALUATE_RERUN = True`, which reruns the pipeline on the 141 validation images;
with it the whole notebook took 64.3 min. The numbers for the 180 test images:

| Item | Value |
| --- | --- |
| Machine | Linux, one NVIDIA RTX A5000 (24 GB), Python 3.11.18 |
| 180 test images | 35.7 min, 11.9 s per image |
| Per image | preprocessing 0.16 s, U-Net 2.48 s, six detectors 6.68 s, fusion + rescorer 0.38 s, refiner 2.20 s |
| Peak GPU memory | about 7 GB (sampled every 10 s) |
| Disk | about 12 GB (environment 6.2 GB, repository, data, outputs) |

Not measured: CPU-only inference (the code falls back to CPU when no GPU is found, but this was never timed or run
end to end), GPUs with less memory, a fresh environment on Windows, and a run without internet access.

## Pipeline

The notebook only calls modules; it has the same five sections as this table.

| Section | Module | What it does |
| --- | --- | --- |
| 1. Preprocessing | `src/solar/final_pipeline.py` | finds the solar disk and removes limb darkening in every image |
| 2. Training (optional) | `src/solar/final_training.py` | prints how each file in `weights/` was produced; `TRAIN = True` retrains instead of loading |
| 3. Inference | `src/solar/final_pipeline.py` | U-Net foreground logits (512 px tiles, 8 flips/rotations); six detectors, four flipped copies each |
| 4. Post-processing | `src/solar/final_pipeline.py` | detections claim U-Net pixels (fusion), the rescorer keeps or drops each candidate, the refiner redraws each kept instance, the CSV is written and checked |
| 5. Evaluation | `src/solar/final_evaluation.py` | PQ, IoU and Dice distributions, one-to-many and many-to-one counts on the 141 held-out images |

Every threshold and setting is fixed in code and listed at the top of the notebook and in each module's docstring.
A second setting, `final2_r0`, uses three of the six detectors; add it to `RUN_FINALS` to rebuild it as well.

## Repository layout

| Path | Content |
| --- | --- |
| `notebooks/final_inference.ipynb` | the demo notebook (start here) |
| `src/solar/` | the three `final_*.py` modules above and the shared code they import (RLE, scoring, submission format) |
| `weights/` | eight model files, two rescorer parameter files, `manifest.json` with SHA-256 and training settings |
| `scripts/` | `verify_final_weights.py` and the dated training scripts that produced the weights; the `final_*.py` modules import or run them, and these scripts import the older dated scripts they build on |
| `tests/` | the four `test_final_*.py` files that cover the shipped pipeline |
| `docs/report/val_predictions/` | the cross-fitted validation predictions read by the evaluation section |
| `docs/experiments/` | the SHA-256 list of the shipped weights, and the reference training arguments read by `scripts/run_yolo11x_detector_20261024.py` |
| `docs/competition/` | the official self-evaluation notebook, with its license and source |
| `third_party/` | vendored code with licenses and provenance (see the last section) |
| `requirements.txt` | the pinned environment |

This repository holds only what the submitted pipeline needs. Experiments that did not lead to the final pipeline
(other architectures, other notebooks, their scripts), the experiment logs and the report figures were left out to
keep it readable. The dated scripts keep the code that produced the weights; a few comments and messages were
reworded for this repository, and comments may still name scripts and records that are not included.

## Reproducibility notes

- **The rebuilt CSV is not byte-identical to the uploaded one.** In the clean-machine run, 1105 of the 1109 uploaded
  instances were reproduced exactly, and 175 of the 178 images with predictions were identical. The three other
  images differ by 2 px, by 5 px, and by one case where two uploaded instances became one. The likely cause is
  that GPU convolutions are not bit-exact across runs and cards; this was not traced layer by layer. The rebuilt CSV
  was not uploaded, so its Public score is unknown.
- **Retraining is not bit-reproducible.** Two of the training recipes set no global seed, cuDNN and mixed precision
  are not bit-exact, and the shipped weights were trained on two GPU types (RTX A5000 and GTX 1080 Ti).
- **The training section (`TRAIN = True`) was run end to end once**, on 2026-11-13, on one RTX A5000 from a fresh
  clone. All 10 steps took 6.7 h and
  used at most about 12.9 GB of GPU memory. The retrained weights then ran inference, post-processing and
  evaluation. On the 141 validation images they score PQ .473, against .474 for the shipped weights; both numbers
  are in-sample for the rescorer and the refiner, and the difference is inside the seed noise (about +-.005). On the
  test images, 1073 of the 1109 uploaded instances (96.8%) have a retrained instance with IoU > .5; none is
  byte-identical. The retrained submission also shows 0.40 on the Public leaderboard. The shipped weights and the
  selected submission were not replaced. Not done: the 3-detector setting's rescorer was not retrained, and
  training was not tried on another GPU model or on CPU.
- `TRAIN_SMOKE = True` (with `TRAIN = True`) runs every training step once with 1-epoch settings, to check that the
  training code runs on a new machine; it took 53 min in the same run and does not produce usable weights.
- `src/solar/final_training.py` calls the same training code with the same arguments as the dated scripts that
  produced the shipped weights; `tests/test_final_training.py` checks this on CPU (same executed source, same
  training arguments, same split).
- **Training needs internet access**: it downloads the two pinned pretrained starting points, and ultralytics
  fetches a font and `yolo11n.pt` for its own mixed-precision check. It writes `hf_home/` and `yolo11n.pt` to the
  repository root (both git-ignored).
- **The U-Net training step needs Linux and an NVIDIA GPU**, because it runs the original training script.
- Rerunning the evaluation with `EVALUATE_RERUN = True` gives PQ .474 on the 141 validation images. That number is
  optimistic: the shipped rescorer and refiner were fitted on those same images. The cross-fitted .467 is the one to
  quote.

## Tests

```bash
python -m pip install pytest==9.1.1
python -m pytest tests/test_final_pipeline.py tests/test_final_training.py tests/test_final_evaluation.py tests/test_final_notebook.py -q
```

Tests that need files outside the repository are skipped when those files are missing. A fresh clone gives 42
passed and 12 skipped; a fresh clone plus the competition data under `data/` gives 48 passed and 6 skipped. The
other 6 tests compare with intermediate files of the original runs (`outputs/`, not in git); with them all 54 pass
on CPU.

## Data, pretrained weights and third-party code

- **Data.** MAGFiLO annotations and GONG H-alpha images as provided by the competition (CC BY-NC 4.0 according to
  the competition Data page). The data is not in this repository.
- **Training labels.** Only the annotations provided by the competition; no extra MAGFiLO annotations.
  Inference reads only the supplied test H-alpha images.
- **Pretrained starting points**, both public and pinned by SHA-256 in [`weights/README.md`](weights/README.md):
  ImageNet EfficientNet-B4 (`smp-hub/efficientnet-b4.imagenet`) for the U-Net and the refiner, and COCO
  `yolo11m-seg.pt` (Ultralytics 8.3.221) for the six detectors. Ultralytics code and models are under AGPL-3.0.
- **Third-party code kept in this repository:**

  | Path | Source | License | Used by the submitted pipeline |
  | --- | --- | --- | --- |
  | `third_party/ameyypawar-filament-seg/` | [ameyypawar/filament-seg](https://github.com/ameyypawar/filament-seg), commit `81a739bb`, unmodified | MIT | yes: disk geometry, flattening, U-Net model and tiled inference; U-Net training loop; training recipe of three detectors |
  | `src/solar/vendor/heshen/train.py` | [HeShen-1/filament-segmentation-2026](https://github.com/HeShen-1/filament-segmentation-2026), commit `b41dae21`, unmodified | MIT | training only: recipe of three detectors |
  | `docs/competition/official_code/` | official self-evaluation notebook by the organizers | Apache-2.0 | reference for the scoring definitions |

  No weights, predictions or data were copied from these sources; every model here was trained from the competition
  annotations.

# weights/: model files and fitted parameters of the two final settings

`notebooks/final_inference.ipynb` loads its weights only from this folder and checks the SHA-256 of every file first.

| File | Role | Size | SHA-256 (first 12) | Output of the original run (under `outputs/`, not in git) |
| --- | --- | --- | --- | --- |
| `unet_effb4_cpstrong_s0.pt` | foreground U-Net (EfficientNet-B4, 2 input channels) | 81.7 MB | `37fe7fb27533` | `unet_cpstrong_20261025/cpstrong_s0/model_last.pt` |
| `yolo11m_seg_cp_s43_1536.pt` | detector cp_s43 (imgsz 1536, raw JPEG) | 45.4 MB | `b1ed8796978e` | `yolo_cp_s43_20261025/det4oof_f1_best.pt` |
| `yolo11m_seg_m1280.pt` | detector m1280 (imgsz 1280, raw JPEG) | 45.3 MB | `63081091d9d1` | `extra_m1280_20261011/det_m1280_f1_best.pt` |
| `yolo11m_seg_m1024.pt` | detector m1024 (imgsz 1024, raw JPEG) | 45.2 MB | `9c16315510e8` | `extra_m1024_20261011/det_m1024_f1_best.pt` |
| `yolo11m_seg_det4_1280.pt` | detector det4 (imgsz 1280, limb-flattened PNG) | 45.3 MB | `c9ac4f030e69` | `det4_amey_20261013/last.pt` |
| `yolo11m_seg_d1024.pt` | detector d1024 (imgsz 1024, limb-flattened PNG) | 45.2 MB | `07c0b58bd02e` | `det_amey_d1024_20261015/last.pt` |
| `yolo11m_seg_d1536.pt` | detector d1536 (imgsz 1536, limb-flattened PNG) | 45.4 MB | `a8cf1185b6a6` | `det_amey_d1536_20261015/last.pt` |
| `refiner_effb4_r0.pt` | instance refiner (EfficientNet-B4, 3 input channels) | 81.7 MB | `313185f892f0` | `refiner_20261031/r_s0_trainALL_r0/model_last.pt` |
| `rescorer_r43.json` | rescorer parameters of `final2_r0` (3-detector pool) | 1.7 KB | — | refitted by `scripts/export_rescorer_params_20261107.py`; equal to the original run's parameters (`combo_cpstrong_s0_cp43_submission_20261026/report.json`) within 1e-9 |
| `rescorer_A6.json` | rescorer parameters of `final1_A6` (6-detector pool) | 1.9 KB | — | same script, compared with `amey_A6_cpstrong_s0_submission_20261104/report.json` |

- Full list with sizes, SHA-256 and training settings: [manifest.json](manifest.json). The pinned SHA-256 list is
  [docs/experiments/final_weights_sha256_20261106.txt](../docs/experiments/final_weights_sha256_20261106.txt).
- Check without PyTorch: `python scripts/verify_final_weights.py` (exit code 0 only when every file matches).

## Pretrained starting points (public general-purpose weights only; no external annotations)

- Encoder of the U-Net and the refiner: ImageNet EfficientNet-B4, `smp-hub/efficientnet-b4.imagenet` revision
  `05cd5dde5dab658f00c463f9b9aa0ced76784f40`, `model.safetensors` SHA-256
  `c465ce05723692769f0d069312aaf8fcdf9a6f5153e48127f5b7eae68e4ec1dd`.
- The six detectors: COCO `yolo11m-seg.pt` (Ultralytics 8.3.221), SHA-256
  `eb9a06f63e2206c35d68d839b08c362429ebecf933ad54c1ad68b2fd001c17cf`.
- Training uses only the MAGFiLO annotations provided by the competition; inference uses only the official test
  H-alpha images.

## How the two final settings use these files

| Setting | Detectors | Rescorer | Kaggle CSV SHA-256 (first 8) |
| --- | --- | --- | --- |
| `final1_A6` (submitted) | cp_s43, m1280, m1024, det4, d1024, d1536 | `rescorer_A6.json` | `2ba22b86` |
| `final2_r0` | cp_s43, m1280, m1024 | `rescorer_r43.json` | `fafef71c` |

The U-Net, the fusion rules and the refiner are the same in both; only the detector set and its rescorer differ.
The switch is `POOLS` / `FINALS` in `src/solar/final_pipeline.py`.

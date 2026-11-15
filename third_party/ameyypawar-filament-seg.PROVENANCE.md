# ameyypawar/filament-seg provenance

Source: https://github.com/ameyypawar/filament-seg at commit
`81a739bb7786a44b989f5d76afbb6f9d22b5d1dd` ("License the code under MIT").
Copyright (c) 2026 Amey Pawar, MIT (see `ameyypawar-filament-seg/LICENSE`).

Vendored unmodified into `third_party/ameyypawar-filament-seg/` on 2026-10-10.
`ameyypawar-filament-seg.sha256` lists the SHA-256 of every vendored file (the
upstream bytes; check with `sha256sum -c` from inside that folder). The
upstream `.git` directory and `reports/` (PDF report and figures) are not kept.
No weights, cached logits, submissions or competition data were copied; all
models used here are retrained from the competition annotations.

Local use:

- Runs in the environment of `requirements.txt` (torch 2.6.0+cu124); upstream
  pinned torch 2.14.0, which is not used.
- `scripts/prepare_amey_unet_fold1.py` writes upstream's `splits.json` from the
  HeShen author fold1 observations instead of upstream `make_splits.py`.
- Data, output and cache paths set through `FILAMENT_DATA_ROOT`,
  `FILAMENT_OUTPUT_ROOT` and `--cache-dir`.

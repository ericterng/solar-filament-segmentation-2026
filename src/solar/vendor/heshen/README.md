# Fixed author source

Unmodified `kernels/det4oof-f1/train.py` from HeShen-1/filament-segmentation-2026,
commit `b41dae215a87c6cb6dfbcf1459a4819021e0092d`.

Source: https://github.com/HeShen-1/filament-segmentation-2026/blob/b41dae215a87c6cb6dfbcf1459a4819021e0092d/kernels/det4oof-f1/train.py

SHA256: `40347ae60cb818ca28798d8e2c57f3f8d412eb4a45b8974778dea32e50f660ee`.
Retrieved 2026-10-10; MIT notice retained in LICENSE. Third-party Ultralytics
code and pretrained YOLO weights have their own AGPL-3.0 terms.

`scripts/run_author_fold.py` verifies the hash and applies checked portability
edits at runtime, saving the complete executed source. The vendor file itself
is never imported as a module because it executes training at top level.

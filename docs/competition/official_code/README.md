# Official self-evaluation notebook

- `self_evaluation.ipynb`: the competition's self-evaluation notebook by Azim Ahmadzadeh (organizer), from
  [Kaggle](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook), Version 6, downloaded 2026-10-05.
  It is kept byte for byte as downloaded (SHA-256 `d4956c0e92cf999dc4417da40f3b61f6e5705e7d596c021ad22cf272507101c8`);
  `.gitattributes` turns off line-ending conversion for it.
- License: Apache-2.0, as stated on the Kaggle page. The license text is in `LICENSE-2.0.txt`.
- Use in this repository: `src/solar/evaluation.py` and `src/solar/final_evaluation.py` apply its definitions
  (strict IoU > 0.5 matching, every annotator record scored separately) to COCO RLE masks. The notebook itself is
  not run by the pipeline.

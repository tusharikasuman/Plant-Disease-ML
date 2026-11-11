## HMM decoding of 40 held-out leaves, compared with the threshold segmentation

| Decoder | Pixel agreement (%) | Lesion recall (%) | Lesion precision (%) | State changes per row |
|---|---|---|---|---|
| Per-pixel colour only (no HMM) | 93.08 | 34.45 | 71.99 | 3.97 |
| Forward-backward (most probable state per pixel) | 95.12 | 97.19 | 37.07 | 2.54 |
| Viterbi (most probable whole path) | 95.12 | 96.99 | 37.10 | 2.48 |
| Threshold segmentation (reference labels) | 100.00 | 100.00 | 100.00 | 2.85 |

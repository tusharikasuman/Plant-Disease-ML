## Coffee SPAD - held-out test on 85 leaves

| Model | R2 | RMSE (SPAD) | MAE (SPAD) |
|---|---|---|---|
| Random forest on colour + HSV histogram (chosen by CV) | 0.91 | 5.51 | 4.23 |
| Paper 2 baseline: least squares on mean R, G, B | 0.86 | 6.65 | 4.79 |
| Always predict the training mean (no model) | -0.00 | 18.04 | 15.69 |

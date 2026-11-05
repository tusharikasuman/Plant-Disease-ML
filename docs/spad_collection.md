# Collecting SPAD readings for the chlorophyll model

The chlorophyll model is trained **only from real SPAD meter readings**. Until `spad/spad_readings.csv`
contains at least 10 real readings (30–50 is much better), the app shows an *uncalibrated greenness
index* and never a SPAD value. Never type in made-up or estimated values.

## What you need
- A chlorophyll meter that reports SPAD units (for example a Konica Minolta SPAD-502 Plus; borrow one from
  an agriculture / botany department). Other meters (e.g. atLEAF) use different units - note which one you used.
- A phone camera.
- A plain white or light-grey sheet (paper or card) as background.
- Leaves of **one crop** (the colour → SPAD relation differs between crops, so calibrate per crop).

## Choosing leaves
- 30–50 leaves (at least 10).
- Cover the **whole range** of greenness: young pale leaves, mature dark-green leaves, older yellowing
  leaves. A model only works inside the range it was trained on.
- Healthy leaves are best. Large lesions make both the SPAD reading and the colour unreliable.

## For each leaf
1. **Photograph it first.** Lay the leaf flat on the background, top side up, filling most of the frame.
   Use the same place, light and distance for every leaf (indirect daylight or a fixed lamp, no flash,
   no strong shadows). One leaf per photo.
2. **Measure it.** Take 3–5 SPAD readings spread over the leaf blade, avoiding the midrib and big veins.
3. **Record** every reading as its own row with the same `image_path`. The program averages readings of the
   same photo automatically.

Put the photos in `spad/photos/` (next to the CSV).

## Filling in `spad/spad_readings.csv`
The file is `spad/spad_readings.csv` and already has these columns:

| column | required | meaning |
|---|---|---|
| `image_path` | yes | path to the photo, relative to the CSV (e.g. `photos/leaf_001.jpg`) or absolute |
| `spad_value` | yes | one SPAD meter reading (number between 0 and 100) |
| `crop` | recommended | e.g. `tomato` |
| `leaf_id` | optional | your own label for the leaf |
| `date` | optional | when it was measured |
| `notes` | optional | anything unusual (e.g. "slightly wilted") |

Example layout (your own numbers go in `spad_value`):

```
image_path,spad_value,crop,leaf_id,date,notes
photos/leaf_001.jpg,<reading 1>,tomato,L1,2026-11-10,
photos/leaf_001.jpg,<reading 2>,tomato,L1,2026-11-10,
photos/leaf_001.jpg,<reading 3>,tomato,L1,2026-11-10,
photos/leaf_002.jpg,<reading 1>,tomato,L2,2026-11-10,
```

## Training
```
python -m src.chlorophyll                 # RGB features (Paper 2's b = A X)
python -m src.chlorophyll --features colour   # RGB + greenness indices
```
This cross-validates OLS, Ridge, Bayesian Ridge, polynomial regression and Random Forest
(leave-one-out CV for up to 60 leaves), saves the best one to `models/chlorophyll_model.joblib`
and writes tables and graphs to `reports/`. Restart the app afterwards: it then shows a calibrated SPAD
value, with a warning when a leaf is outside the SPAD range you measured.

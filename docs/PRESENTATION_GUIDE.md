# How to explain this project (speaking guide)

Use this as your script. Each part says **what to show**, **what to say**, and **what they may ask**.
Total time ≈ 12–15 minutes. Graphs are in `reports/figures/15_…` to `19_…`.

---

## Part 1 — The 1-minute story (say this first)

> "Farmers need to know early if a plant is sick. A phone photo is the cheapest sensor there is.
> Most systems use deep learning, which is a black box and needs a GPU. We did the opposite: we turn
> every leaf photo into **55 numbers a human can understand** — colour, greenness, how much of the leaf
> is brown, texture and shape — and then use the classical machine-learning models from our syllabus on
> those numbers. It runs on a normal laptop and every prediction can be explained.
> On 11,628 tomato leaf photos in 5 classes, our best model is **96.3% accurate** on leaves it never saw.
> We also estimate **chlorophyll** from the photo, trained on real chlorophyll-meter readings of coffee
> leaves, with R² = 0.91."

Three questions the system answers about one photo:
1. Is the leaf diseased? 2. Which disease? 3. How much of the leaf is damaged (and how green is it)?

---

## Part 2 — How it works (the pipeline, 2 minutes)

Draw or show this:

```
photo → ① clean → ② find the leaf → ③ 55 numbers → ④ ML models → ⑤ answer + explanation
```

| Step | What we do | Why (say this) |
|---|---|---|
| ① Clean | resize to 256×256, small blur | every photo the same size; removes camera noise |
| ② Find the leaf | convert to HSV colour; green pixels = leaf tissue, brown pixels = lesions; fill gaps, keep biggest region, remove shadows | features must come from the **leaf only**, so the background cannot fool the model |
| ③ 55 features | colour averages (6), greenness + lesion % (4), colour histogram (32), texture (6), shape (7) | each is one meaningful number — e.g. lesion %: healthy 1%, late blight 12.6% |
| ④ Models | every model = "scale the numbers → model"; tuned with 5-fold cross-validation; tested **once** on 2,326 held-out leaves | fair comparison, no cheating |
| ⑤ Output | disease, confidence, lesion %, top-3, "why this prediction", "refer to expert" if unsure | explainable and safe |

**Key honest point:** the 2,326 test leaves are never used for training or choosing a model.

---

## Part 3 — How the units connect (the "alignment", 2 minutes)

Say: *"Each unit answers one question about the same leaf data, and each one builds on the one before."*

| Unit | Question it answers | What we used | Headline result |
|---|---|---|---|
| **1** | Can simple probability (Bayes' rule) diagnose a leaf? | Gaussian Naive Bayes, correlation/VIF analysis | 71.1% — weak, because features are correlated (R and G: 0.91), which breaks Naive Bayes' assumption |
| **2** | Do linear models do better? Can the model say when it's unsure? | Logistic regression, LDA, SVM (linear + RBF), Bayesian logistic regression (from scratch) | RBF SVM 93.3%; Bayesian LR 99.5% healthy vs diseased, and it knows when it is unsure |
| **3** | Is there structure without labels? How many dimensions do we really need? | PCA, k-means, Gaussian mixture + EM | 30 of 55 dimensions keep 95% of the information; clusters only partly match diseases → labels are needed |
| **4** | Can we use the **order** of pixels? | Hidden Markov model on each pixel row; forward–backward, Viterbi | cleaner lesion maps: 95.1% agreement, fewer noisy jumps (2.48 vs 3.97 per row) |
| **5** | Can advanced models beat Unit 2, and give honest error bars? | Neural network, combining models, kernel methods, sampling & graphical models | Gradient boosting 96.3% (best); neural net 93.2%; GP error bars; chlorophyll R² 0.91 |

**The thread that ties Unit 5 back to earlier units (say these links out loud):**
- Neural network = logistic regression (Unit 2) **plus a hidden layer** → it compares itself to LR.
- Boosting/committees combine the Unit 1 and Unit 2 models.
- Kernel methods reuse the **SVM** from Unit 2 and explain *why* it works (support vectors).
- MCMC **checks** the Laplace approximation we used in Unit 2's Bayesian LR.
- Gibbs sampling in an MRF is the **2-D version of the Unit 4 HMM** ("neighbouring pixels agree").

---

## Part 4 — The Unit 5 graphs, one by one

For each graph: **What it shows → How to read it → What to say → Likely question.**

### Topic A — Neural networks (`15_neural_networks`)

What we built: a network written by us in NumPy: **55 inputs → 32 hidden units (tanh) → 5 outputs (softmax)**,
trained with **backpropagation** and the Adam optimiser. We checked our gradients against numerical
derivatives (error 2 × 10⁻⁶ = correct).

**Graph 1 — `1_training_curves.png`**
- *Shows:* left = error (cross-entropy) per epoch; right = accuracy per epoch; blue = training leaves,
  orange = held-out leaves.
- *Read it:* both errors fall fast in the first 10 epochs; after ~25 epochs the orange (held-out) line goes
  flat at about 0.2 while blue keeps falling. Accuracy: training ends ≈ 98%, held-out ≈ 93%.
- *Say:* "Backpropagation works — the error drops smoothly. The gap between blue and orange is the start of
  over-fitting: after about 25 epochs the network is memorising training leaves, not learning more."
- *Q: Why not stop earlier?* — Held-out accuracy is flat, not falling, so 60 epochs does no harm; weight
  decay keeps the gap small.

**Graph 2 — `2_accuracy_vs_hidden_units.png`**
- *Shows:* accuracy as we give the network 1, 2, 4 … 128 hidden units.
- *Read it:* with 1 unit only 67%; it rises fast to ~91% at 8 units; after 32 units held-out stays ≈ 94%
  while training climbs to 99.7%.
- *Say:* "More hidden units = more flexible model. Up to 32 units it helps both lines; after that only
  training accuracy improves — that widening gap is **over-fitting**. That's why we chose 32."

**Graph 3 — `3_decision_boundaries_2d.png`**
- *Shows:* leaves squeezed onto 2 dimensions (PCA, from Unit 3); background colour = which disease the model
  would predict in that area. Left: logistic regression; right: neural network.
- *Read it:* logistic regression can only draw **straight lines**; the network draws **curved regions** that
  follow the clusters (e.g. the healthy group bottom-left). On these 2-D points: 45.8% vs 56.0%.
- *Say:* "This is *why* the hidden layer helps: it learns curved boundaries. With all 55 features the
  network gets 93.2% vs 89.3% for logistic regression."
- *Q: Why is accuracy only ~50% here?* — Only 2 of 55 dimensions are used, just for the picture.

### Topic B — Combining models (`16_combining_models`)

**Graph 4 — `1_adaboost_rounds.png`**
- *What we built:* AdaBoost from scratch with **decision stumps** (a stump asks one yes/no question about one
  feature).
- *Left:* error vs number of stumps. Starts at ~48% (one stump), falls to ~16% at 300 stumps. The dotted black
  line is **sklearn's AdaBoost** — it lies exactly on ours (same predictions on 100% of leaves).
- *Right:* ε (each stump's weighted error, grey) rises and α (its voting power, green) falls.
- *Say:* "Each new stump focuses on the leaves the previous ones got wrong, because those leaves get higher
  weights. Later stumps face harder leaves, so their error goes up and their vote counts less. A stump alone
  is 52%; 300 of them together reach 84%."
- *Q: Error above 50% — isn't that worse than guessing?* — With 5 classes, guessing is 80% error. SAMME allows
  error up to 80% and still gives a positive vote (α = ln((1−ε)/ε) + ln 4).

**Graph 5 — `2_committee_vs_members.png`**
- *Shows:* grey = five single models; blue = committees; orange = AdaBoost; dashed line = average member.
- *Read it:* every committee beats the average member (83.6%). Hard vote 89.6%, soft vote 88.7%.
  **Stacking 91.8% beats every single member.**
- *Say:* "Averaging cancels independent mistakes — committee error is never worse than the average member's.
  A plain vote is dragged down by weak Naive Bayes; stacking *learns* how much to trust each model."
- *Q: Then which is the best overall?* — Gradient boosting (96.3%), also a combining method — trees added one
  by one, each correcting the last (Topic B in the syllabus too).

### Topic C — Kernel methods (`17_kernel_methods`)

**Graph 6 — `1_svm_support_vectors_vs_C.png`**
- *Task:* healthy vs diseased with an RBF-kernel SVM on 1,500 leaves.
- *Read it:* blue = % of leaves that are support vectors; orange = accuracy. As C grows (0.01 → 1000)
  support vectors fall from 41% to under 10%, while accuracy rises from 86% to 99%.
- *Say:* "An SVM is a **sparse** kernel machine: the final decision uses only the support vectors. We rebuilt
  the decision function by hand from them — it matches sklearn to 10⁻¹³ using just 162 of 1,500 leaves. The
  other 1,338 leaves could be thrown away."

**Graph 7 — `2_gaussian_process_regression.png`**
- *Task:* predict disease **severity** (√lesion %) from colour and texture only.
- *Left:* GP on one input. Blue line = prediction; shaded band = 95% interval. The band is **wider where there
  is little data** (far left/right).
- *Right:* all 400 test leaves sorted; orange dots (true values) mostly inside the blue error bars.
- *Say:* "A Gaussian process gives every prediction an **error bar**. 94% of test leaves fall inside the 95%
  band, so the error bars are honest. R² = 0.84."

**Graph 8 — `3_rvm_vs_svr_sparsity.png`**
- *Left:* our Relevance Vector Machine (from scratch): predictions vs truth with error bars, close to the
  diagonal (R² 0.76).
- *Right:* how many training leaves each model keeps: GP 600, SVR 447, **RVM 35**.
- *Say:* "The RVM is a Bayesian sparse model: it switches off almost all training points and keeps 35 —
  13× sparser than SVR — and still gives error bars."

### Topic D — Sampling and graphical models (`18_sampling_graphical_models`)

**Graph 9 — `1_mcmc_vs_laplace.png`**
- *Left:* Metropolis–Hastings traces of 3 weights over 30,000 steps; grey band = burn-in (thrown away);
  dashed line = the Laplace/MAP answer. Traces wander around the dashed line → the chain has mixed.
- *Right:* each blue dot is one posterior sample; orange ellipses = the Laplace Gaussian (1 and 2 std).
  The dots fill the ellipses.
- *Say:* "In Unit 2 we *approximated* the posterior with a Gaussian (Laplace). Here we draw samples from the
  **true** posterior. They agree — so the Unit 2 approximation was justified."

**Graph 10 — `2_gibbs_mrf_lesion_denoising.png`**
- *Read left to right:* photo → true lesion mask → we flip 15% of pixels at random (noise) → Gibbs posterior
  P(lesion) → cleaned mask with **2.1%** wrong pixels.
- *Say:* "An Ising Markov random field says 'neighbouring pixels usually agree'. Gibbs sampling updates one
  pixel at a time given its neighbours. It removes the noise but keeps the real spots. This is the 2-D
  version of our Unit 4 HMM, which only looked along rows."

**Graph 11 — `3_bayes_net_inference.png`**
- *Left:* our Bayesian network: Disease → Lesion level, Yellowing, Texture; Lesion → Texture.
  p(D,L,Y,T) = p(D) p(L|D) p(Y|D) p(T|D,L). Orange = observed (texture = high).
- *Middle:* P(disease | texture high): exact answer (black) vs 3 samplers. Dashed boxes = prior. E.g. Yellow
  leaf curl drops from ~46% prior to 24%; healthy rises from ~14% to 25% (healthy leaves have strong veins).
- *Right:* sampler error vs number of samples on log–log axes, following the dotted 1/√N line.
- *Say:* "Exact inference sums over all combinations. Rejection sampling, likelihood weighting and Gibbs
  sampling all approach the exact answer, and their error shrinks like 1/√N — 100× more samples, 10× less
  error."

### Extra — Chlorophyll (`19_chlorophyll_spad`)

**`1_predicted_vs_measured.png`**
- *Left:* each dot = a held-out coffee leaf; x = SPAD meter, y = our prediction from the photo. Close to the
  diagonal; R² 0.91, error 5.5 SPAD. Grey bars = meter's own spread between its 5 readings.
- *Right:* residuals scattered around 0 with no pattern → no systematic error.
- *Say:* "Tomato photos have no chlorophyll readings, so we used a public coffee dataset with real meter
  readings. Random forest beats the published linear baseline (R² 0.86 → 0.91). The meter itself varies by
  3.2 SPAD, so 5.5 is close to the limit."
- `2_model_comparison.png`: all models × feature sets by CV error — non-linear models win; plain least squares
  on all 55 collinear features fails (R² 0.54).
- `3_greenness_vs_spad_and_error_bars.png`: greenness follows chlorophyll but is not in SPAD units; the GP
  gives each leaf an error bar, and on a tomato leaf the bar becomes 4× wider → flagged as untrustworthy.

---

## Part 5 — Final comparison slide (1 minute)

Same 2,326 test leaves for every model:

```
Naive Bayes 71.1 → LDA 85.0 → Logistic Reg 89.3 → Stacking 91.8 → Neural net 93.2
→ RBF SVM 93.3 → Random forest 94.1 → Gradient boosting 96.3 (best)
```

Say: "Accuracy climbs as models become able to draw curved boundaries and combine many learners."

---

## Part 6 — Limitations (say these yourself; it earns marks)

1. Trained on **lab photos** (one leaf, plain background). In our testing, a warm camera colour tint dropped
   accuracy to 33%, and a brown background was counted as disease.
2. It always names one of 5 tomato classes — a non-tomato photo still gets a confident answer.
3. Early blight is the hardest class (82.5% recall), confused with late blight.
4. Chlorophyll is calibrated for **coffee only**.

Future work: field photos, colour calibration (white-balance card), an "is this a tomato leaf?" check,
SPAD readings for tomato, mobile app.

---

## Part 7 — Likely questions and short answers

| Question | Answer |
|---|---|
| Why not deep learning? | Our course is classical ML; our features are explainable, run on a CPU, and still reach 96%. |
| Why is test accuracy (96.3) higher than CV (92.2)? | CV used a balanced sample; the test set has many easy yellow-leaf-curl leaves. Balanced accuracy (94.1) removes that effect. |
| How do you know your from-scratch code is correct? | Each one is checked: backprop vs numerical gradients, AdaBoost vs sklearn (100% same), SVM dual vs sklearn (10⁻¹³), MCMC vs Laplace; 102 automated tests. |
| What is over-fitting, from your graphs? | The gap between training and held-out curves (graphs 1 and 2). |
| Bagging vs boosting? | Bagging (random forest) averages independent trees → less variance. Boosting adds trees one after another, each fixing previous errors → less bias. |
| What does the Bayesian model add? | An honest confidence. Wrong answers average 69% confidence, right ones 99%; sending the least-sure 20% to an expert leaves 100% accuracy on the rest. |
| What is a support vector? | A training leaf on or inside the margin; only these define the SVM decision. |
| Why does Naive Bayes lose? | It assumes features are independent, but R and G are 91% correlated. |
| How is Unit 4 "sequential" data? | Each row of pixels is read left to right like a sentence; hidden state = background / healthy / lesion. |
| Where is the prediction explained? | The "Why this prediction?" panel: logistic-regression weight × feature value, summed per feature group. |

---

## Part 8 — Live demo (optional, 2 minutes)

```
python demo.py --step 6        # neural network, live
python -m src.predict samples/tomato_early_blight.jpg
streamlit run app.py           # pick "tomato early blight": show mask, top-3, "why"
```

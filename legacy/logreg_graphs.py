"""
logreg_graphs.py - every standard Logistic Regression graph on our 1,500 real PlantVillage tomato leaves.
Input : data.npz produced by ml_unit1_unit2_proof.py (53 features per leaf, 5 classes, lesion-area %)
Output: fig/lr_*.png (one per graph) + fig/LR_all_graphs.png (all on one sheet) + printed metrics
"""
import numpy as np, json, glob, warnings, sys
SHOW = '--show' in sys.argv          # python logreg_graphs.py --show  -> also open each graph in a live window
import matplotlib
if not SHOW: matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, learning_curve, validation_curve
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_fscore_support,
                             accuracy_score, log_loss, classification_report)
warnings.filterwarnings('ignore')
for f in glob.glob('/usr/share/fonts/truetype/google-fonts/*.ttf'): fm.fontManager.addfont(f)
INK, MUTE, GRID = '#1F2A22', '#6B7266', '#E6E9E4'
PAL = ['#2A9160', '#C9760E', '#3867C9', '#CC3F7A', '#8A6A12']   # validated categorical palette
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': '#BDB6A4',
                     'axes.labelcolor': INK, 'xtick.color': MUTE, 'ytick.color': MUTE, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlelocation': 'left',
                     'axes.titlesize': 11.5, 'axes.titleweight': 'bold'})

d = np.load('data.npz', allow_pickle=True)
X, y, L, names = d['X'], d['y'], d['L'], list(d['names'])
short = ['Healthy', 'Early blight', 'Late blight', 'Septoria', 'YLCV']
feat = ['R mean', 'G mean', 'B mean', 'R std', 'G std', 'B std', 'GLI', 'G/R'] + \
       [f'HSV h{h} s{s} v{v}' for h in range(8) for s in range(2) for v in range(2)] + \
       ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM'] + [f'Hu {i}' for i in range(1, 8)]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
LR = lambda C=1.0: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
out = {}

# ---------------- multi-class, all 53 features ----------------
proba = cross_val_predict(LR(), X, y, cv=cv, method='predict_proba')
pred = proba.argmax(1)
out['acc_5class'] = accuracy_score(y, pred); out['logloss_5class'] = log_loss(y, proba)
print('=== 5-class Logistic Regression, 53 features, 5-fold CV ===')
print(f'accuracy {out["acc_5class"]*100:.2f}%   log-loss {out["logloss_5class"]:.3f}')
print(classification_report(y, pred, target_names=short, digits=3))

# ---------------- binary: healthy vs diseased ----------------
yb = (y != 0).astype(int)                                # 1 = diseased
xb = np.log10(L * 100 + 0.5).reshape(-1, 1)              # one feature: brown area (log %)
lr1 = LogisticRegression(class_weight='balanced').fit(xb, yb)   # balanced = equal priors, else it just predicts the 80% majority
w, b0 = lr1.coef_[0, 0], lr1.intercept_[0]
x50 = -b0 / w
out['binary1_w'] = w; out['binary1_b'] = b0; out['binary1_boundary_pct'] = 10 ** x50 - 0.5
pb_all = cross_val_predict(LR(), X, yb, cv=cv, method='predict_proba')[:, 1]
out['acc_binary_53'] = accuracy_score(yb, pb_all > .5)
from sklearn.metrics import balanced_accuracy_score
pb_1 = cross_val_predict(LogisticRegression(class_weight='balanced'), xb, yb, cv=cv, method='predict_proba')[:, 1]
out['acc_binary_1'] = balanced_accuracy_score(yb, pb_1 > .5)
print(f'=== Binary healthy vs diseased ===\n1 feature (brown %): w={w:.3f}, b={b0:.3f}, P=0.5 at {out["binary1_boundary_pct"]:.2f}% brown, CV balanced acc {out["acc_binary_1"]*100:.2f}%')
print(f'all 53 features: CV acc {out["acc_binary_53"]*100:.2f}%  (balanced {balanced_accuracy_score(yb, pb_all > .5)*100:.2f}%)')

fig_list = []
def save(fig, name): fig.savefig(f'fig/{name}.png', dpi=180, facecolor='white'); fig_list.append(name); plt.close(fig)

def p_sigmoid(ax):
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.04, 0.04, len(yb))
    ax.scatter(xb[yb == 0, 0], yb[yb == 0] + jit[yb == 0], s=7, color=PAL[0], alpha=.45, lw=0, label='Healthy (y = 0)')
    ax.scatter(xb[yb == 1, 0], yb[yb == 1] + jit[yb == 1], s=7, color=PAL[1], alpha=.35, lw=0, label='Diseased (y = 1)')
    xs = np.linspace(xb.min() - .1, xb.max() + .1, 400)
    ax.plot(xs, 1 / (1 + np.exp(-(w * xs + b0))), color=INK, lw=2.2)
    ax.axhline(.5, color=GRID, lw=1); ax.axvline(x50, color=MUTE, ls=':', lw=1)
    ax.text(x50 + .06, .40, f'P = 0.5 at\n{10**x50-0.5:.1f}% brown', fontsize=8.5, color=MUTE)
    ax.text(xs[0] + .02, .80, f'P(diseased) = 1 / (1 + e^-({w:.2f}·x {b0:+.2f}))', fontsize=8.5)
    tk = [0, 1, 2, 5, 10, 25, 100]; ax.set_xticks([np.log10(t + .5) for t in tk]); ax.set_xticklabels(tk)
    ax.set_xlabel('x = brown area, % of leaf (log scale)'); ax.set_ylabel('P(diseased)')
    ax.set_title('1 · Sigmoid curve (healthy vs diseased, one feature)')
    ax.text(xs[-1], .2, f'balanced accuracy {out["acc_binary_1"]*100:.0f}% with 1 feature\nvs {out["acc_binary_53"]*100:.0f}% with all 53', fontsize=8, color=MUTE, ha='right'); ax.legend(frameon=False, fontsize=8, loc='lower right', markerscale=2)

# 2-D view: two real features (mean Blue, GLI) - the best pair we found, still only ~50%
from sklearn.model_selection import cross_val_score
sc2 = StandardScaler().fit(X[:, [2, 6]]); Z = sc2.transform(X[:, [2, 6]])
lr2 = LogisticRegression(max_iter=5000).fit(Z, y)
out['acc_2feat_cv'] = cross_val_score(LR(), X[:, [2, 6]], y, cv=cv).mean()
def p_regions(ax):
    gx, gy = np.meshgrid(np.linspace(Z[:, 0].min() - 1, Z[:, 0].max() + 1, 400), np.linspace(Z[:, 1].min() - 1, Z[:, 1].max() + 1, 400))
    G = lr2.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contourf(gx, gy, G, levels=np.arange(-.5, 5), colors=[c + '26' for c in PAL])
    ax.contour(gx, gy, G, levels=np.arange(.5, 4), colors=INK, linewidths=.7)
    for k in range(5): ax.scatter(Z[y == k, 0], Z[y == k, 1], s=6, color=PAL[k], lw=0, alpha=.75, label=short[k])
    ax.set_xlabel('mean Blue (standardised)'); ax.set_ylabel('GLI greenness (standardised)')
    ax.set_title(f'2 · Decision regions, 2 features only ({out["acc_2feat_cv"]*100:.0f}% CV)'); ax.legend(frameon=False, fontsize=7.5, markerscale=2.5, loc='upper right')
    ax.text(.02, .02, 'straight boundaries = linear classifier.\nTwo features overlap too much; 53 get 80%', transform=ax.transAxes, fontsize=8, color=MUTE)

def p_cm(ax):
    cm = confusion_matrix(y, pred); cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap=ListedColormap(plt.cm.Greens(np.linspace(.03, .95, 256))), vmin=0, vmax=1)
    for i in range(5):
        for j in range(5): ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=9.5, color='white' if cmn[i, j] > .55 else INK)
    lab = ['Healthy', 'Early\nblight', 'Late\nblight', 'Septoria', 'YLCV']
    ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=8); ax.set_yticks(range(5)); ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(f'3 · Confusion matrix (accuracy {out["acc_5class"]*100:.1f}%)')

Yb = label_binarize(y, classes=range(5)); out['auc'] = {}
def p_roc(ax):
    for k in range(5):
        fpr, tpr, _ = roc_curve(Yb[:, k], proba[:, k]); a = auc(fpr, tpr); out['auc'][short[k]] = a
        ax.plot(fpr, tpr, color=PAL[k], lw=2, label=f'{short[k]}  AUC {a:.3f}')
    ax.plot([0, 1], [0, 1], color=GRID, ls='--', lw=1); ax.text(.55, .42, 'random guess', color=MUTE, fontsize=8, rotation=38)
    ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_title('4 · ROC curves (one class vs rest)'); ax.legend(frameon=False, fontsize=8, loc='lower right')

pr, rc, f1, _ = precision_recall_fscore_support(y, pred)
def p_prf(ax):
    xx = np.arange(5); wd = .26
    for i, (v, lab, c) in enumerate([(pr, 'Precision', '#1F2A22'), (rc, 'Recall', PAL[0]), (f1, 'F1', '#9FC9AE')]):
        ax.bar(xx + (i - 1) * wd, v * 100, wd - .03, color=c, label=lab)
    for k in range(5): ax.text(k, max(pr[k], rc[k], f1[k]) * 100 + 1.5, f'{f1[k]*100:.0f}', ha='center', fontsize=8.5, fontweight='bold')
    ax.set_xticks(xx); ax.set_xticklabels(short, fontsize=8.5); ax.set_ylim(50, 105); ax.set_ylabel('%')
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.0))
    ax.set_title('5 · Precision, recall, F1 per class (F1 labelled)')

Cs = np.logspace(-3, 3, 13)
tr, va = validation_curve(LR(), X, y, param_name='logisticregression__C', param_range=Cs, cv=cv)
bestC = Cs[va.mean(1).argmax()]; out['bestC'] = bestC; out['bestC_acc'] = va.mean(1).max()
def p_C(ax):
    ax.plot(Cs, tr.mean(1) * 100, color=MUTE, lw=2, marker='o', ms=4, label='training accuracy')
    ax.plot(Cs, va.mean(1) * 100, color=PAL[0], lw=2, marker='o', ms=4, label='validation accuracy (5-fold)')
    ax.fill_between(Cs, (va.mean(1) - va.std(1)) * 100, (va.mean(1) + va.std(1)) * 100, color=PAL[0], alpha=.15, lw=0)
    ax.axvline(bestC, color=MUTE, ls=':', lw=1); ax.text(bestC * 1.3, 52, f'best C ≈ {bestC:.1f}\n{va.mean(1).max()*100:.1f}%', fontsize=8.5, color=MUTE)
    ax.set_xscale('log'); ax.set_xlabel('C = 1/λ   (← stronger regularisation · weaker →)'); ax.set_ylabel('accuracy (%)'); ax.set_ylim(40, 100)
    ax.text(1.2e-3, 92, 'underfits', fontsize=8, color=MUTE); ax.text(900, 93.5, 'train–validation gap', fontsize=8, color=MUTE, ha='right')
    ax.set_title('6 · Regularisation strength vs accuracy'); ax.legend(frameon=False, fontsize=8, loc='lower right')

sizes, ltr, lva = learning_curve(LR(), X, y, cv=cv, train_sizes=np.linspace(.08, 1, 9), shuffle=True, random_state=0)
def p_learn(ax):
    ax.plot(sizes, ltr.mean(1) * 100, color=MUTE, lw=2, marker='o', ms=4, label='training accuracy')
    ax.plot(sizes, lva.mean(1) * 100, color=PAL[0], lw=2, marker='o', ms=4, label='validation accuracy')
    ax.fill_between(sizes, (lva.mean(1) - lva.std(1)) * 100, (lva.mean(1) + lva.std(1)) * 100, color=PAL[0], alpha=.15, lw=0)
    ax.set_xlabel('number of training leaves'); ax.set_ylabel('accuracy (%)'); ax.set_ylim(40, 101)
    ax.text(sizes[-1], lva.mean(1)[-1] * 100 - 6, 'still rising →\nmore data would help', fontsize=8, color=MUTE, ha='right')
    ax.set_title('7 · Learning curve'); ax.legend(frameon=False, fontsize=8, loc='lower right')

full = LR().fit(X, y); coef = full[-1].coef_          # standardised weights, shape 5 x 53
def p_coef(ax):
    c = coef[0]; idx = np.argsort(np.abs(c))[-12:]
    cols = [PAL[0] if v > 0 else PAL[1] for v in c[idx]]
    ax.barh(range(12), c[idx], color=cols, height=.65); ax.axvline(0, color=INK, lw=.8)
    ax.set_yticks(range(12)); ax.set_yticklabels([feat[i] for i in idx], fontsize=8); ax.tick_params(axis='y', length=0)
    ax.set_xlabel('weight (standardised features)')
    ax.text(.98, .04, 'green = pushes towards Healthy\norange = pushes away', transform=ax.transAxes, ha='right', fontsize=8, color=MUTE)
    ax.set_title('8 · Top 12 weights for the "Healthy" class')

conf = proba.max(1); ok = pred == y
def p_conf(ax):
    bins = np.linspace(.2, 1, 33)
    ax.hist(conf[ok], bins=bins, color=PAL[0], alpha=.8, label=f'correct ({ok.sum()})')
    ax.hist(conf[~ok], bins=bins, color=PAL[1], alpha=.85, label=f'wrong ({(~ok).sum()})')
    ax.set_xlabel('model confidence = highest predicted probability'); ax.set_ylabel('number of leaves')
    ax.set_title('9 · Confidence of right vs wrong predictions'); ax.legend(frameon=False, fontsize=8, loc='upper left')
    out['mean_conf_correct'] = conf[ok].mean(); out['mean_conf_wrong'] = conf[~ok].mean()

panels = [('lr_1_sigmoid', p_sigmoid), ('lr_2_decision_regions', p_regions), ('lr_3_confusion', p_cm), ('lr_4_roc', p_roc),
          ('lr_5_precision_recall_f1', p_prf), ('lr_6_regularisation_C', p_C), ('lr_7_learning_curve', p_learn),
          ('lr_8_weights', p_coef), ('lr_9_confidence', p_conf)]
for name, fn in panels:
    fig, ax = plt.subplots(figsize=(6.4, 4.6)); fn(ax); fig.tight_layout(); save(fig, name)

fig, axs = plt.subplots(3, 3, figsize=(19, 16.5))
for ax, (name, fn) in zip(axs.ravel(), panels): fn(ax)
fig.suptitle('Logistic Regression on 1,500 real PlantVillage tomato leaves  ·  53 features  ·  5-fold stratified CV',
             x=.01, ha='left', fontsize=16, fontweight='bold', y=.995)
fig.tight_layout(rect=(0, 0, 1, .975), h_pad=3, w_pad=2.5); fig.savefig('fig/LR_all_graphs.png', dpi=130, facecolor='white'); plt.close(fig)

print('=== extra numbers ===')
print('AUC per class:', {k: round(v, 3) for k, v in out['auc'].items()})
print(f'best C = {bestC:.1f}  (CV accuracy {out["bestC_acc"]*100:.2f}%)')
print(f'learning curve: {sizes[0]} leaves → {lva.mean(1)[0]*100:.1f}%,  {sizes[-1]} leaves → {lva.mean(1)[-1]*100:.1f}%')
print(f'2 features (B mean, GLI) 5-class CV accuracy: {out["acc_2feat_cv"]*100:.1f}%')
print(f'mean confidence: correct {out["mean_conf_correct"]:.2f}, wrong {out["mean_conf_wrong"]:.2f}')
print('top Healthy weights:', [(feat[i], round(coef[0, i], 2)) for i in np.argsort(np.abs(coef[0]))[::-1][:6]])
json.dump({k: (v if not isinstance(v, np.floating) else float(v)) for k, v in out.items()}, open('lr_results.json', 'w'), indent=1, default=float)

if SHOW:   # one live window at a time: close it to see the next graph
    for name, fn in panels:
        fig, ax = plt.subplots(figsize=(9, 6)); fn(ax); fig.tight_layout()
        fig.canvas.manager.set_window_title(name)
        plt.show()

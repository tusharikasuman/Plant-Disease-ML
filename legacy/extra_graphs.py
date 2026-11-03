"""
extra_graphs.py - 12 more graphs on the same 1,500 PlantVillage tomato leaves.
Input : data.npz produced by ml_unit1_unit2_proof.py (53 features per leaf, 5 classes, lesion-area %)
Output: fig/ex_*.png (one per graph) + fig/EX_all_graphs.png (all on one sheet) + printed numbers
Usage : python extra_graphs.py            (save the PNGs)
        python extra_graphs.py --show     (also open each graph in a live window; close it to see the next)
        In Colab / Jupyter: run it in a cell after data.npz exists - graphs appear under the cell.
"""
import numpy as np, os, sys, warnings
try:
    get_ipython(); NOTEBOOK = True          # running inside Colab / Jupyter
except NameError:
    NOTEBOOK = False
SHOW = NOTEBOOK or '--show' in sys.argv
import matplotlib
if not SHOW: matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, r2_score, mean_absolute_error
from sklearn.calibration import calibration_curve
warnings.filterwarnings('ignore')
INK, MUTE, GRID = '#1F2A22', '#6B7266', '#E6E9E4'
PAL = ['#2A9160', '#C9760E', '#3867C9', '#CC3F7A', '#8A6A12']   # same palette as logreg_graphs.py
MARK = ['o', 's', '^', 'D', 'v']                                  # a shape per class too, not colour alone
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': '#BDB6A4',
                     'axes.labelcolor': INK, 'xtick.color': MUTE, 'ytick.color': MUTE, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlelocation': 'left',
                     'axes.titlesize': 11.5, 'axes.titleweight': 'bold'})
os.makedirs('fig', exist_ok=True)

d = np.load('data.npz', allow_pickle=True)
X, y, L = d['X'], d['y'], d['L']
short = ['Healthy', 'Early blight', 'Late blight', 'Septoria', 'YLCV']
feat = ['R mean', 'G mean', 'B mean', 'R std', 'G std', 'B std', 'GLI', 'G/R'] + \
       [f'HSV h{h} s{s} v{v}' for h in range(8) for s in range(2) for v in range(2)] + \
       ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM'] + [f'Hu {i}' for i in range(1, 8)]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
sc = lambda m: make_pipeline(StandardScaler(), m)
print('computing ... (about 1-2 minutes)')

# ---------------- numbers behind the graphs ----------------
models = {'Logistic Regression': LogisticRegression(max_iter=5000), 'LDA': LinearDiscriminantAnalysis(),
          'SVM (linear)': SVC(kernel='linear'), 'SVM (RBF)': SVC(kernel='rbf', C=10),
          'KNN (k=5)': KNeighborsClassifier(5), 'Random Forest': RandomForestClassifier(300, random_state=0)}
folds, f1s = {}, {}
for n, m in models.items():
    folds[n] = cross_val_score(sc(m), X, y, cv=cv) * 100
    f1s[n] = f1_score(y, cross_val_predict(sc(m), X, y, cv=cv), average=None) * 100
print('\n=== accuracy of each model in each of the 5 folds (this is where mean ± sd comes from) ===')
for n, s in folds.items(): print(f'{n:20s} folds {np.round(s, 1)}  ->  {s.mean():.2f} ± {s.std():.2f}')

rf = RandomForestClassifier(300, random_state=0).fit(X, y)
ks = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
k_acc = np.array([cross_val_score(sc(KNeighborsClassifier(k)), X, y, cv=cv) * 100 for k in ks])
trees = [5, 10, 25, 50, 100, 200, 300]
t_acc = np.array([cross_val_score(RandomForestClassifier(t, random_state=0), X, y, cv=cv) * 100 for t in trees])
Cs, gammas = [0.1, 1, 10, 100], [0.001, 0.01, 0.1, 1]
grid = np.array([[cross_val_score(sc(SVC(C=c, gamma=g)), X, y, cv=cv).mean() * 100 for g in gammas] for c in Cs])
p_lr = cross_val_predict(sc(LogisticRegression(max_iter=5000)), X, y, cv=cv, method='predict_proba')
p_rf = cross_val_predict(RandomForestClassifier(300, random_state=0), X, y, cv=cv, method='predict_proba')
Z = PCA(2).fit(StandardScaler().fit_transform(X)); P = Z.transform(StandardScaler().fit_transform(X))

# regression: predict brown-lesion % of the leaf from 8 simple colour numbers (RGB stats, GLI, G/R)
Xr, yr = X[:, :8], L * 100
kf = KFold(5, shuffle=True, random_state=0)
reg = {'Linear': sc(LinearRegression()), 'Ridge': sc(Ridge(10))}
reg_pred = {n: cross_val_predict(m, Xr, yr, cv=kf) for n, m in reg.items()}
print('\n=== regression: brown lesion % from 8 colour features (5-fold CV) ===')
for n, p in reg_pred.items(): print(f'{n:7s} R² {r2_score(yr, p):.3f}   mean abs error {mean_absolute_error(yr, p):.2f} % points')
print(f'\nKNN best k = {ks[k_acc.mean(1).argmax()]} ({k_acc.mean(1).max():.1f}%)')
print(f'Random Forest: {trees[0]} trees {t_acc.mean(1)[0]:.1f}%  ->  {trees[-1]} trees {t_acc.mean(1)[-1]:.1f}%')
i, j = np.unravel_index(grid.argmax(), grid.shape)
print(f'SVM best C = {Cs[i]}, gamma = {gammas[j]}  ({grid.max():.1f}%)')
print(f'PCA: first 2 components explain {Z.explained_variance_ratio_.sum()*100:.1f}% of the variation')

# ---------------- the graphs ----------------
def g_folds(ax):
    names = list(folds)[::-1]
    for r, n in enumerate(names):
        s = folds[n]
        ax.scatter(s, np.full(5, r), s=30, color=MUTE, alpha=.55, zorder=2, lw=0)
        ax.errorbar(s.mean(), r, xerr=s.std(), fmt='o', color=PAL[0], ms=9, capsize=5, lw=2, zorder=3)
        ax.text(s.max() + .6, r, f'{s.mean():.1f} ± {s.std():.1f}', va='center', fontsize=8.5)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.set_xlim(70, 95)
    ax.set_xlabel('accuracy (%)  -  grey dots = the 5 folds, green = mean ± sd')
    ax.set_title('1 · Where "mean ± sd" comes from: 5 CV folds per model')

def g_f1(ax):
    M = np.array(list(f1s.values()))
    ax.imshow(M, cmap='Greens', vmin=50, vmax=100, aspect='auto')
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            ax.text(c, r, f'{M[r, c]:.0f}', ha='center', va='center', fontsize=9, color='white' if M[r, c] > 82 else INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(short, fontsize=8.5); ax.set_yticks(range(len(f1s))); ax.set_yticklabels(list(f1s))
    ax.spines[:].set_visible(False); ax.set_title('2 · F1 score (%) per disease, per model')

def g_pca(ax):
    for k in range(5):
        ax.scatter(P[y == k, 0], P[y == k, 1], s=12, color=PAL[k], marker=MARK[k], alpha=.55, lw=0, label=short[k])
    ev = Z.explained_variance_ratio_ * 100
    ax.set_xlabel(f'PC 1 ({ev[0]:.0f}% of variation)'); ax.set_ylabel(f'PC 2 ({ev[1]:.0f}% of variation)')
    ax.legend(frameon=False, fontsize=8, markerscale=1.8); ax.set_title('3 · PCA: all 53 features squeezed into 2')

def g_imp(ax):
    top = np.argsort(rf.feature_importances_)[::-1][:15][::-1]
    ax.barh(range(15), rf.feature_importances_[top] * 100, color=PAL[0], height=.7)
    ax.set_yticks(range(15)); ax.set_yticklabels([feat[t] for t in top], fontsize=8.5)
    ax.set_xlabel('importance (%)'); ax.set_title('4 · Random Forest: top 15 most useful features')

def g_corr(ax):
    C = np.corrcoef(X.T)
    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
    for a, lab in [(0, 'RGB'), (8, 'HSV histogram'), (40, 'texture'), (46, 'shape')]:
        ax.axhline(a - .5, color='white', lw=1.5); ax.axvline(a - .5, color='white', lw=1.5)
        ax.text(a, -1.5, lab, fontsize=8, color=MUTE)
    ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    plt.colorbar(im, ax=ax, fraction=.046, pad=.02, label='correlation')
    ax.set_title('5 · Correlation between the 53 features', pad=16)

def g_lesion(ax):
    data = [np.log10(L[y == k] * 100 + .5) for k in range(5)]
    bp = ax.boxplot(data, widths=.55, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5))
    for b, c in zip(bp['boxes'], PAL): b.set(facecolor=c, alpha=.35, edgecolor=c)
    rng = np.random.default_rng(0)
    for k in range(5): ax.scatter(k + 1 + rng.uniform(-.18, .18, len(data[k])), data[k], s=4, color=PAL[k], alpha=.35, lw=0)
    tk = [0, 1, 2, 5, 10, 25, 50]; ax.set_yticks([np.log10(t + .5) for t in tk]); ax.set_yticklabels(tk)
    ax.set_xticks(range(1, 6)); ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylabel('brown area, % of leaf (log scale)'); ax.set_title('6 · How much of each leaf is brown, by class')

def g_knn(ax):
    m, s = k_acc.mean(1), k_acc.std(1)
    ax.fill_between(ks, m - s, m + s, color=PAL[0], alpha=.15, lw=0); ax.plot(ks, m, '-o', color=PAL[0], lw=2, ms=6)
    b = m.argmax(); ax.annotate(f'best k = {ks[b]}\n{m[b]:.1f}%', (ks[b], m[b]), (ks[b] + 8, m[b] + 1.5), fontsize=8.5,
                                arrowprops=dict(arrowstyle='-', color=MUTE))
    ax.set_xlabel('k = number of neighbours'); ax.set_ylabel('CV accuracy (%)'); ax.set_title('7 · KNN: choosing k')

def g_trees(ax):
    m, s = t_acc.mean(1), t_acc.std(1)
    ax.fill_between(trees, m - s, m + s, color=PAL[0], alpha=.15, lw=0); ax.plot(trees, m, '-o', color=PAL[0], lw=2, ms=6)
    ax.set_xscale('log'); ax.set_xticks(trees); ax.set_xticklabels(trees)
    ax.set_xlabel('number of trees'); ax.set_ylabel('CV accuracy (%)'); ax.set_title('8 · Random Forest: more trees, then it levels off')

def g_svm(ax):
    ax.imshow(grid, cmap='Greens', vmin=grid.min() - 5, vmax=100, aspect='auto')
    for r in range(len(Cs)):
        for c in range(len(gammas)):
            ax.text(c, r, f'{grid[r, c]:.1f}', ha='center', va='center', fontsize=9.5,
                    color='white' if grid[r, c] > 80 else INK, fontweight='bold' if (r, c) == (i, j) else 'normal')
    ax.set_xticks(range(len(gammas))); ax.set_xticklabels(gammas); ax.set_yticks(range(len(Cs))); ax.set_yticklabels(Cs)
    ax.set_xlabel('gamma (how local each support vector is)'); ax.set_ylabel('C (1/regularisation)')
    ax.spines[:].set_visible(False); ax.set_title('9 · SVM (RBF): accuracy for each C and gamma')

def g_pr(ax):
    Y = label_binarize(y, classes=range(5))
    for k in range(5):
        p, r, _ = precision_recall_curve(Y[:, k], p_lr[:, k])
        ax.plot(r, p, color=PAL[k], lw=2, label=f'{short[k]}  AP {average_precision_score(Y[:, k], p_lr[:, k]):.2f}')
    ax.axhline(.2, color=GRID, ls='--', lw=1); ax.text(.5, .22, 'random guess (20%)', ha='center', fontsize=8, color=MUTE)
    ax.set_xlabel('recall'); ax.set_ylabel('precision'); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(0, .25)); ax.set_title('10 · Precision-recall curves (Logistic Regression)')

def g_calib(ax):
    for n, p, c in [('Logistic Regression', p_lr, PAL[0]), ('Random Forest', p_rf, PAL[2])]:
        conf, ok = p.max(1), p.argmax(1) == y
        fr, mp = calibration_curve(ok, conf, n_bins=8, strategy='quantile')
        ax.plot(mp, fr, '-o', color=c, lw=2, ms=6, label=n)
    ax.plot([0, 1], [0, 1], ls='--', color=MUTE, lw=1); ax.text(.62, .55, 'perfectly honest', rotation=38, fontsize=8, color=MUTE)
    ax.set_xlim(.2, 1); ax.set_ylim(.2, 1.02)
    ax.set_xlabel('how sure the model says it is'); ax.set_ylabel('how often it is actually right')
    ax.legend(frameon=False, fontsize=8, loc='upper left'); ax.set_title('11 · Calibration: is the model\'s confidence honest?')

def g_reg(ax):
    p = reg_pred['Ridge']
    for k in range(5): ax.scatter(yr[y == k], p[y == k], s=10, color=PAL[k], marker=MARK[k], alpha=.5, lw=0, label=short[k])
    lim = [min(yr.min(), p.min()) - 1, max(yr.max(), p.max()) + 1]
    ax.plot(lim, lim, ls='--', color=MUTE, lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.text(.03, .95, f'Ridge regression, 8 colour features\nR² = {r2_score(yr, p):.2f}   '
                      f'mean error = {mean_absolute_error(yr, p):.1f} % points', transform=ax.transAxes, va='top', fontsize=8.5)
    ax.set_xlabel('actual brown area (%)'); ax.set_ylabel('predicted brown area (%)')
    ax.legend(frameon=False, fontsize=8, loc='lower right', markerscale=1.8); ax.set_title('12 · Regression: predicting how brown a leaf is')

panels = [('ex_1_cv_folds', g_folds), ('ex_2_f1_heatmap', g_f1), ('ex_3_pca', g_pca), ('ex_4_rf_importance', g_imp),
          ('ex_5_feature_correlation', g_corr), ('ex_6_lesion_by_class', g_lesion), ('ex_7_knn_k', g_knn),
          ('ex_8_rf_trees', g_trees), ('ex_9_svm_grid', g_svm), ('ex_10_precision_recall', g_pr),
          ('ex_11_calibration', g_calib), ('ex_12_regression', g_reg)]
for name, fn in panels:
    fig, ax = plt.subplots(figsize=(7.5, 5.2)); fn(ax); fig.tight_layout()
    fig.savefig(f'fig/{name}.png', dpi=180, facecolor='white'); plt.close(fig)

fig, axs = plt.subplots(4, 3, figsize=(20, 22))
for ax, (name, fn) in zip(axs.ravel(), panels): fn(ax)
fig.suptitle('More graphs on 1,500 real PlantVillage tomato leaves  ·  53 features  ·  5-fold CV',
             x=.01, ha='left', fontsize=16, fontweight='bold', y=.997)
fig.tight_layout(rect=(0, 0, 1, .98), h_pad=3, w_pad=2.5); fig.savefig('fig/EX_all_graphs.png', dpi=120, facecolor='white'); plt.close(fig)
print(f'\nsaved {len(panels)} graphs to fig/ex_*.png and fig/EX_all_graphs.png')

if SHOW:   # script: one live window at a time (close it to see the next); notebook: all appear under the cell
    for name, fn in panels:
        fig, ax = plt.subplots(figsize=(9, 6)); fn(ax); fig.tight_layout()
        if not NOTEBOOK: fig.canvas.manager.set_window_title(name)
        plt.show()

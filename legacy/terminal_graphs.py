"""
terminal_graphs.py - draws the project's graphs directly inside the terminal (no pictures, no windows).
Input : data.npz produced by ml_unit1_unit2_proof.py
Usage : python terminal_graphs.py            (press Enter to go to the next graph)
        python terminal_graphs.py --nopause  (print all graphs one after another)
Needs : pip install plotext
"""
import sys, warnings
import numpy as np
import plotext as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict, learning_curve, validation_curve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, r2_score
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')             # so the Windows terminal can print the chart characters
PAUSE = '--nopause' not in sys.argv

d = np.load('data.npz', allow_pickle=True)
X, y, L = d['X'], d['y'], d['L']
short = ['Healthy', 'Early blight', 'Late blight', 'Septoria', 'YLCV']
COL = [(42, 145, 96), (201, 118, 14), (56, 103, 201), (204, 63, 122), (138, 106, 18)]   # same colours as the PNG graphs
GREEN = COL[0]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
sc = lambda m: make_pipeline(StandardScaler(), m)
LR = lambda C=1.0: sc(LogisticRegression(C=C, max_iter=5000))

def start(title, xlabel='', ylabel=''):
    plt.clf(); plt.theme('pro'); plt.plotsize(min(plt.terminal_width(), 110), 28)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)

def finish(note='', draw=True):
    if draw: plt.show()
    if note: print('  ' + note)
    if PAUSE: input('\n  Press Enter for the next graph ... ')
    print()

def g_models():
    print('  training 6 models with 5-fold cross-validation ...')
    models = {'Logistic Regression': LogisticRegression(max_iter=5000), 'LDA': LinearDiscriminantAnalysis(),
              'SVM (linear)': SVC(kernel='linear'), 'SVM (RBF)': SVC(kernel='rbf', C=10),
              'KNN (k=5)': KNeighborsClassifier(5), 'Random Forest': RandomForestClassifier(300, random_state=0)}
    acc = {n: cross_val_score(sc(m), X, y, cv=cv) * 100 for n, m in models.items()}
    start('1. Accuracy of each model (5-fold CV, 1,500 tomato leaves)', 'accuracy (%)')
    plt.bar(list(acc), [round(s.mean(), 1) for s in acc.values()], orientation='horizontal', color=GREEN, width=.6)
    plt.xlim(0, 100)
    finish('mean ± sd:  ' + '   '.join(f'{n} {s.mean():.1f} ± {s.std():.1f}' for n, s in acc.items()))

def g_sigmoid():
    yb = (y != 0).astype(int); xb = np.log10(L * 100 + .5).reshape(-1, 1)
    m = LogisticRegression(class_weight='balanced').fit(xb, yb); w, b = m.coef_[0, 0], m.intercept_[0]
    start('2. Sigmoid curve: P(diseased) from brown area', 'brown area, log10(% of leaf + 0.5)', 'P(diseased)')
    rng = np.random.default_rng(0); j = rng.uniform(-.04, .04, len(yb))
    plt.scatter(xb[yb == 0, 0], (yb + j)[yb == 0], color=COL[0], label='Healthy (y = 0)', marker='dot')
    plt.scatter(xb[yb == 1, 0], (yb + j)[yb == 1], color=COL[1], label='Diseased (y = 1)', marker='dot')
    xs = np.linspace(xb.min(), xb.max(), 200)
    plt.plot(xs, 1 / (1 + np.exp(-(w * xs + b))), color='white', label='sigmoid', marker='braille')
    finish(f'P(diseased) = 1 / (1 + e^-({w:.2f}x {b:+.2f}))   ->   P = 0.5 at {10 ** (-b / w) - .5:.1f}% brown')

def g_confusion():
    pred = cross_val_predict(LR(), X, y, cv=cv); cm = confusion_matrix(y, pred)
    print('\n  3. Confusion matrix - Logistic Regression (rows = true class, columns = predicted)\n')
    print(' ' * 16 + ''.join(f'{s:>14s}' for s in short))
    for i, row in enumerate(cm):
        cells = ''.join((f'\033[1;32m{v:>13d}\033[0m' if i == k else f'{v:>13d}') for k, v in enumerate(row))
        print(f'  {short[i]:14s}{cells}')
    finish(f'accuracy = {np.trace(cm) / cm.sum() * 100:.1f}%   (green = correct guesses on the diagonal)', draw=False)

def g_roc():
    p = cross_val_predict(LR(), X, y, cv=cv, method='predict_proba'); Y = label_binarize(y, classes=range(5))
    start('4. ROC curves, one class vs rest (Logistic Regression)', 'false positive rate', 'true positive rate')
    for k in range(5):
        f, t, _ = roc_curve(Y[:, k], p[:, k]); plt.plot(f, t, color=COL[k], label=f'{short[k]} AUC {auc(f, t):.3f}', marker='braille')
    plt.plot([0, 1], [0, 1], color='gray', label='random guess')
    finish('AUC 1.0 = perfect, 0.5 = guessing')

def g_learning():
    sizes, tr, va = learning_curve(LR(), X, y, cv=cv, train_sizes=np.linspace(.08, 1, 8))
    start('5. Learning curve (Logistic Regression)', 'number of training leaves', 'accuracy (%)')
    plt.plot(sizes, tr.mean(1) * 100, color='gray', label='training accuracy', marker='hd')
    plt.plot(sizes, va.mean(1) * 100, color=GREEN, label='validation accuracy', marker='hd')
    finish(f'{sizes[0]} leaves -> {va.mean(1)[0] * 100:.1f}%,   {sizes[-1]} leaves -> {va.mean(1)[-1] * 100:.1f}%   (still rising: more data would help)')

def g_C():
    Cs = np.logspace(-3, 3, 13)
    tr, va = validation_curve(LR(), X, y, param_name='logisticregression__C', param_range=Cs, cv=cv)
    start('6. Regularisation strength C vs accuracy', 'log10(C)   (left = strong regularisation)', 'accuracy (%)')
    plt.plot(np.log10(Cs), tr.mean(1) * 100, color='gray', label='training', marker='hd')
    plt.plot(np.log10(Cs), va.mean(1) * 100, color=GREEN, label='validation (5-fold)', marker='hd')
    finish(f'best C = {Cs[va.mean(1).argmax()]:.1f}  ({va.mean(1).max() * 100:.1f}%)')

def g_ablation():
    colour = list(range(40)); tex = list(range(40, 46)); hu = list(range(46, 53))
    sets = [('RGB stats only', list(range(6))), ('Colour', colour), ('Colour + texture', colour + tex), ('+ shape', colour + tex + hu)]
    acc = [cross_val_score(sc(SVC(C=10)), X[:, idx], y, cv=cv).mean() * 100 for _, idx in sets]
    start('7. Which features matter? (SVM RBF accuracy)', 'accuracy (%)')
    plt.bar([s for s, _ in sets], [round(a, 1) for a in acc], orientation='horizontal', color=GREEN, width=.6); plt.xlim(0, 100)
    finish('texture adds ~3 points on top of colour; shape adds nothing')

def g_knn():
    ks = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
    acc = [cross_val_score(sc(KNeighborsClassifier(k)), X, y, cv=cv).mean() * 100 for k in ks]
    start('8. KNN: accuracy for each k', 'k = number of neighbours', 'accuracy (%)')
    plt.plot(ks, acc, color=GREEN, marker='hd')
    finish(f'best k = {ks[int(np.argmax(acc))]}  ({max(acc):.1f}%)')

def g_brown():
    start('9. Average brown area of the leaf, by class', '', 'brown area (% of leaf)')
    plt.bar(short, [round(L[y == k].mean() * 100, 1) for k in range(5)], color=COL, width=.6)
    finish('healthy leaves have the least brown, blight leaves the most')

def g_pca():
    P = PCA(2).fit_transform(StandardScaler().fit_transform(X))
    start('10. PCA: all 53 features squeezed into 2 dimensions', 'PC 1', 'PC 2')
    for k in range(5): plt.scatter(P[y == k, 0], P[y == k, 1], color=COL[k], label=short[k], marker='dot')
    finish('the classes overlap a lot in 2-D, which is why we need all 53 features')

def g_regression():
    yr = L * 100; p = cross_val_predict(sc(Ridge(10)), X[:, :8], yr, cv=KFold(5, shuffle=True, random_state=0))
    start('11. Regression: predicted vs actual brown area (Ridge)', 'actual brown area (%)', 'predicted brown area (%)')
    plt.scatter(yr, p, color=GREEN, marker='dot', label='leaves')
    plt.plot([0, yr.max()], [0, yr.max()], color='gray', label='perfect prediction')
    finish(f'R² = {r2_score(yr, p):.2f}  (1.0 = perfect)   mean error = {np.abs(yr - p).mean():.1f} % points')

graphs = [g_models, g_sigmoid, g_confusion, g_roc, g_learning, g_C, g_ablation, g_knn, g_brown, g_pca, g_regression]
print(f'\n  Plant disease ML project - {len(graphs)} graphs drawn in the terminal  ({len(y)} leaves, {X.shape[1]} features)\n')
for g in graphs: g()
print('  done.')

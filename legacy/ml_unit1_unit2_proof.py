"""
ml_unit1_unit2_proof.py
Reproduces every "our output" number and figure in PlantDisease_Unit1_Unit2_Mapping.pptx (slides 9-14).

Usage:  python ml_unit1_unit2_proof.py /path/to/PlantVillage/raw/color
        (folder that contains Tomato___healthy, Tomato___Early_blight, ... sub-folders;
         the Kaggle "New Plant Diseases Dataset" train/ folder works too if the class folder names match)
Needs:  pip install opencv-python scikit-image scikit-learn scipy matplotlib numpy
Outputs: data.npz, res.json, u1.json, cm.npy and fig/*.png in the current folder.
"""
import sys, os
os.makedirs('fig', exist_ok=True)
import sys
import numpy as np, cv2, glob, os, random, json
from skimage.feature import graycomatrix, graycoprops
random.seed(7); np.random.seed(7)
ROOT=sys.argv[1] if len(sys.argv)>1 else 'raw/color'
CLASSES={'Tomato___healthy':'Healthy','Tomato___Early_blight':'Early blight','Tomato___Late_blight':'Late blight',
 'Tomato___Septoria_leaf_spot':'Septoria leaf spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus':'Yellow leaf curl virus'}
N=300
def leaf_mask(img):
    # Paper 1 style: green + brown threshold masks in HSV, union
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    green=cv2.inRange(hsv,(25,40,40),(95,255,255))
    brown=cv2.inRange(hsv,(5,40,20),(25,255,255))
    m=cv2.bitwise_or(green,brown)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    return m,green,brown
def feats(path):
    img=cv2.imread(path); img=cv2.resize(img,(256,256))
    img=cv2.GaussianBlur(img,(3,3),0)
    m,green,brown=leaf_mask(img)
    px=img[m>0].astype(float)  # BGR
    B,G,R=px[:,0],px[:,1],px[:,2]
    rgb=[R.mean(),G.mean(),B.mean(),R.std(),G.std(),B.std()]
    gli=((2*G-R-B)/(2*G+R+B+1e-6)).mean()
    gr=(G/(R+1e-6)).mean()
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1,2],m,[8,2,2],[0,180,0,256,0,256]).flatten(); hist/=hist.sum()+1e-9
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); gray[m==0]=0
    g=graycomatrix((gray//8).astype(np.uint8),[1,2],[0,np.pi/4,np.pi/2,3*np.pi/4],levels=32,symmetric=True,normed=True)
    har=[graycoprops(g,p).mean() for p in ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']]
    hu=cv2.HuMoments(cv2.moments(m)).flatten(); hu=-np.sign(hu)*np.log10(np.abs(hu)+1e-30)
    lesion=(brown>0).sum()/max((m>0).sum(),1)
    return rgb+[gli,gr]+list(hist)+har+list(hu), lesion
X=[];y=[];L=[];paths=[]
for k,(d,name) in enumerate(CLASSES.items()):
    fs=sorted(glob.glob(f'{ROOT}/{d}/*')); fs=random.sample(fs,N)
    for f in fs:
        v,les=feats(f); X.append(v); y.append(k); L.append(les); paths.append(f)
np.savez('data.npz',X=np.array(X),y=np.array(y),L=np.array(L),paths=np.array(paths),names=np.array(list(CLASSES.values())))
print(np.array(X).shape)

# ================= MODELS (slides 11-14) =================
import numpy as np, json
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
d=np.load('data.npz'); X,y=d['X'],d['y']
cv=StratifiedKFold(5,shuffle=True,random_state=0)
models={'Logistic Regression':LogisticRegression(max_iter=5000,C=1.0),
 'LDA':LinearDiscriminantAnalysis(),
 'SVM (linear kernel)':SVC(kernel='linear',C=1),
 'SVM (RBF kernel)':SVC(kernel='rbf',C=10,gamma='scale'),
 'KNN (k=5)':KNeighborsClassifier(5),
 'Random Forest':RandomForestClassifier(300,random_state=0)}
res={}
for n,m in models.items():
    p=make_pipeline(StandardScaler(),m); s=cross_val_score(p,X,y,cv=cv)
    res[n]=(s.mean()*100,s.std()*100); print(f'{n:22s} {s.mean()*100:.2f} ± {s.std()*100:.2f}')
# colour-only vs colour+texture for SVM RBF
colour=list(range(0,8))+list(range(8,40)); tex=list(range(40,46)); hu=list(range(46,53))
for lab,idx in [('RGB stats only',list(range(0,6))),('Colour (RGB+indices+HSV hist)',colour),('Colour + texture',colour+tex),('Colour + texture + shape',colour+tex+hu)]:
    s=cross_val_score(make_pipeline(StandardScaler(),SVC(kernel='rbf',C=10)),X[:,idx],y,cv=cv); res['abl_'+lab]=(s.mean()*100,s.std()*100); print(lab,round(s.mean()*100,2))
pred=cross_val_predict(make_pipeline(StandardScaler(),SVC(kernel='rbf',C=10)),X,y,cv=cv)
cm=confusion_matrix(y,pred); np.save('cm.npy',cm); print(cm)
json.dump(res,open('res.json','w'),indent=1)
# Unit1: correlation of mean R,G,B
M=X[:,0:3]; C=np.corrcoef(M.T); print('corr',C.round(3)); print('cov',np.cov(M.T).round(1))
# VIF
vif=[]
for j in range(3):
    o=[k for k in range(3) if k!=j]; r2=LinearRegression().fit(M[:,o],M[:,j]).score(M[:,o],M[:,j]); vif.append(1/(1-r2))
print('VIF',np.round(vif,2)); json.dump({'corr':C.tolist(),'vif':vif},open('u1.json','w'))

# ================= FIGURES =================
import numpy as np, cv2, json, glob, random
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

BG='#FFFFFF'; INK='#1F2A22'; MUTE='#6B7266'; GRID='#E2DDCF'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.facecolor':BG,'figure.facecolor':BG,'savefig.facecolor':BG,
 'axes.edgecolor':'#BDB6A4','axes.labelcolor':INK,'xtick.color':MUTE,'ytick.color':MUTE,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':False,'text.color':INK})
PAL=['#2A9160','#C9760E','#3867C9','#CC3F7A','#8A6A12']
d=np.load('data.npz',allow_pickle=True); X,y,L,paths,names=d['X'],d['y'],d['L'],d['paths'],list(d['names'])
res=json.load(open('res.json'))
def rgb(p): return cv2.cvtColor(cv2.resize(cv2.imread(p),(256,256)),cv2.COLOR_BGR2RGB)
def mask(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    m=cv2.bitwise_or(cv2.inRange(hsv,(25,40,40),(95,255,255)),cv2.inRange(hsv,(5,40,20),(25,255,255)))
    return cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
# pick representative images
random.seed(3)
pick={k:paths[y==k][[5,40,77,120]] for k in range(5)}
# ---- Fig 1: densities healthy vs early blight
fig,axs=plt.subplots(2,2,figsize=(9,5.6),gridspec_kw={'width_ratios':[1,2.2]})
for r,k in enumerate([0,1]):
    im=rgb(pick[k][0]); m=mask(im)
    axs[r,0].imshow(im); axs[r,0].axis('off'); axs[r,0].set_title(names[k],fontsize=11,loc='left',color=INK)
    ax=axs[r,1]; px=im[m>0]
    for c,(col,lab) in enumerate([('#B8423A','R'),('#2A9160','G'),('#3867C9','B')]):
        h,e=np.histogram(px[:,c],bins=64,range=(0,256),density=True)
        ax.plot((e[:-1]+e[1:])/2,h,color=col,lw=2)
        mu,sd=px[:,c].mean(),px[:,c].std()
        ax.axvline(mu,color=col,lw=1,ls=':')
        ax.text(0.99,0.95-c*0.13,f'{lab}   mean μ = {mu:.0f}    std σ = {sd:.0f}',color=col,fontsize=9.5,ha='right',va='top',transform=ax.transAxes,fontweight='bold')
    ax.set_yticks([]); ax.set_xlim(0,255)
    if r==1: ax.set_xlabel('pixel intensity (0–255), leaf pixels only')
plt.tight_layout(); plt.savefig('fig/u1_density.png',dpi=200); plt.close()
# ---- Fig 2: covariance scatter R vs G
fig,ax=plt.subplots(figsize=(6.2,5))
for k in range(5):
    s=y==k; ax.scatter(X[s,0],X[s,1],s=10,color=PAL[k],alpha=.55,edgecolors='none',label=names[k])
C=json.load(open('u1.json'))
ax.set_xlabel('mean Red per leaf'); ax.set_ylabel('mean Green per leaf')
ax.text(0.03,0.95,f'r(R,G) = {C["corr"][0][1]:.2f}\nVIF(R) = {C["vif"][0]:.1f}',transform=ax.transAxes,va='top',fontsize=13,fontweight='bold')
lg=ax.legend(frameon=False,fontsize=8.5,loc='lower right',markerscale=2)
ax.set_title('1,500 tomato leaves — every dot is one image',loc='left',fontsize=11,color=MUTE)
plt.tight_layout(); plt.savefig('fig/u1_cov.png',dpi=200); plt.close()
# ---- Fig 3: Bayes on lesion fraction
from scipy.stats import norm
lx=np.log10(L*100+0.5); h=lx[y==0]; dz=lx[y!=0]
xs=np.linspace(lx.min()-0.1,lx.max()+0.1,500)
p_h=(y==0).mean(); p_d=1-p_h
lh=norm.pdf(xs,h.mean(),h.std()); ld=norm.pdf(xs,dz.mean(),dz.std())
post=ld*p_d/(lh*p_h+ld*p_d); post_eq=ld/(lh+ld)
fig,(a1,a2)=plt.subplots(2,1,figsize=(7,5.6),sharex=True,gridspec_kw={'height_ratios':[1.4,1]})
bins=np.linspace(xs[0],xs[-1],45)
a1.hist(h,bins=bins,density=True,color=PAL[0],alpha=.3); a1.hist(dz,bins=bins,density=True,color=PAL[1],alpha=.3)
a1.plot(xs,lh,color=PAL[0],lw=2); a1.plot(xs,ld,color=PAL[1],lw=2)
a1.text(h.mean()-0.05,lh.max()*1.04,'Healthy   p(x | H)',ha='center',fontsize=9.5,color=INK)
a1.text(dz.mean()+0.3,ld.max()*1.1,'Diseased   p(x | D)',ha='left',fontsize=9.5,color=INK)
hm=max(np.histogram(h,bins=bins,density=True)[0].max(),np.histogram(dz,bins=bins,density=True)[0].max()); a1.set_yticks([]); a1.set_ylim(0,hm*1.08)
a1.set_title('Likelihoods fitted as Gaussians on 1,500 leaves',loc='left',fontsize=10,color=MUTE)
a2.plot(xs,post_eq,color=INK,lw=2); a2.plot(xs,post,color=PAL[1],lw=2,ls='--')
a2.axhline(.5,color=GRID,lw=1)
i=[j for j in np.where(np.diff(np.sign(post_eq-.5)))[0] if h.mean()-0.3<xs[j]<dz.mean()+0.3]; xb=xs[i[0]]
a2.axvline(xb,color=MUTE,ls=':',lw=1)
a2.text(xb+0.04,0.06,f'boundary (equal priors)\n≈ {10**xb-0.5:.1f}% brown area',fontsize=8.5,color=MUTE)
a2.text(xs[-1],0.93,'priors 0.2 / 0.8',color=PAL[1],fontsize=9,ha='right',va='top')
a2.text(xs[-1],0.62,'equal priors 0.5 / 0.5',color=INK,fontsize=9,ha='right',va='top')
a2.set_ylabel('P(Diseased | x)'); a2.set_ylim(0,1.05)
tk=[0,0.5,1,2,5,10,25,50,100]; a2.set_xticks([np.log10(t+0.5) for t in tk]); a2.set_xticklabels([f'{t}' for t in tk])
a2.set_xlabel('x = brown / lesion pixels as % of leaf area (log scale)')
plt.tight_layout(); plt.savefig('fig/u1_bayes.png',dpi=200); plt.close()
print('bayes boundary %',10**xb-0.5, 'H mean%',(10**h.mean()-0.5),'D mean%',10**dz.mean()-0.5)
# ---- Fig 4: paper vs ours dumbbell
paper={'Logistic Regression':90.45,'LDA':88.77,'KNN':93.20,'Random Forest':96.62,'SVM':92.98}
ours={'Logistic Regression':res['Logistic Regression'][0],'LDA':res['LDA'][0],'KNN':res['KNN (k=5)'][0],'Random Forest':res['Random Forest'][0],'SVM':res['SVM (RBF kernel)'][0]}
order=sorted(paper,key=lambda k:paper[k])
fig,ax=plt.subplots(figsize=(7.2,4.2))
for i,k in enumerate(order):
    ax.plot([ours[k],paper[k]],[i,i],color=GRID,lw=3,zorder=1)
    ax.scatter(paper[k],i,s=90,color=INK,zorder=3); ax.scatter(ours[k],i,s=90,color=PAL[0],zorder=3)
    ax.text(paper[k]+0.6,i,f'{paper[k]:.1f}',va='center',fontsize=9.5)
    ax.text(ours[k]-0.6,i,f'{ours[k]:.1f}',va='center',ha='right',fontsize=9.5,color=PAL[0])
ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.set_xlim(68,101)
ax.set_xlabel('accuracy (%)'); ax.tick_params(axis='y',length=0); ax.spines['left'].set_visible(False)
ax.scatter([],[],s=60,color=INK,label='Kiran & Chandrappa 2023 — 40 classes, 41,546 images, tuned'); ax.scatter([],[],s=60,color=PAL[0],label='Our re-run — 5 tomato classes, 1,500 images, untuned')
ax.legend(frameon=False,fontsize=8.5,loc='upper center',bbox_to_anchor=(0.45,1.2),ncol=1)
plt.tight_layout(); plt.savefig('fig/u2_models.png',dpi=200); plt.close()
# ---- Fig 5: ablation
labs=['RGB stats only\n(our poster today)','+ vegetation indices\n+ HSV histogram','+ Haralick texture','+ Hu shape moments']
keys=['abl_RGB stats only','abl_Colour (RGB+indices+HSV hist)','abl_Colour + texture','abl_Colour + texture + shape']
v=[res[k][0] for k in keys]; e=[res[k][1] for k in keys]
fig,ax=plt.subplots(figsize=(7.4,4))
cols=[ '#BDB6A4',PAL[0],PAL[0],PAL[0]]
b=ax.bar(range(4),v,color=cols,width=.58,yerr=e,error_kw=dict(ecolor=MUTE,lw=1,capsize=3))
for i,val in enumerate(v): ax.text(i,val+2.2,f'{val:.1f}%',ha='center',fontsize=11,fontweight='bold')
ax.set_xticks(range(4)); ax.set_xticklabels(labs,fontsize=9); ax.set_ylim(50,100); ax.set_ylabel('5-fold CV accuracy (%)')
ax.set_title('Same SVM (RBF kernel) — only the feature set changes',loc='left',fontsize=10.5,color=MUTE)
plt.tight_layout(); plt.savefig('fig/u2_ablation.png',dpi=200); plt.close()
# ---- Fig 6: kernel trick linear vs RBF + confusion
cm=np.load('cm.npy'); cmn=cm/cm.sum(1,keepdims=True)
fig,ax=plt.subplots(figsize=(5.6,4.8))
from matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('g',[BG,'#2A9160','#14452D'])
ax.imshow(cmn,cmap=cmap,vmin=0,vmax=1)
short=['Healthy','Early\nblight','Late\nblight','Septoria','YLCV']
for i in range(5):
    for j in range(5):
        ax.text(j,i,cm[i,j],ha='center',va='center',fontsize=10,color='white' if cmn[i,j]>.5 else INK)
ax.set_xticks(range(5)); ax.set_xticklabels(short,fontsize=8.5); ax.set_yticks(range(5)); ax.set_yticklabels(short,fontsize=8.5)
ax.set_xlabel('predicted'); ax.set_ylabel('true'); [s.set_visible(False) for s in ax.spines.values()]
ax.tick_params(length=0)
plt.tight_layout(); plt.savefig('fig/u2_cm.png',dpi=200); plt.close()
# ---- Fig 7: OLS vs Ridge coefficient stability (bootstrap, n=30)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
M=StandardScaler().fit_transform(X[:,0:3]); t=L
rng=np.random.default_rng(1); co=[];cr=[]
for _ in range(300):
    idx=rng.choice(len(t),30,replace=False)
    co.append(LinearRegression().fit(M[idx],t[idx]).coef_); cr.append(Ridge(alpha=10).fit(M[idx],t[idx]).coef_)
co=np.array(co); cr=np.array(cr)
np.save('ridge_sd.npy',np.array([co.std(0),cr.std(0)]))
fig,ax=plt.subplots(figsize=(7,4))
pos=np.arange(3)
bp1=ax.boxplot([co[:,j] for j in range(3)],positions=pos-0.18,widths=.3,patch_artist=True,showfliers=False)
bp2=ax.boxplot([cr[:,j] for j in range(3)],positions=pos+0.18,widths=.3,patch_artist=True,showfliers=False)
for bp,c in [(bp1,'#BDB6A4'),(bp2,PAL[0])]:
    for p in bp['boxes']: p.set_facecolor(c); p.set_edgecolor(INK); p.set_linewidth(.8)
    for k in ['whiskers','caps','medians']:
        for l in bp[k]: l.set_color(INK); l.set_linewidth(.8)
ax.axhline(0,color=GRID,lw=1)
ax.set_xticks(pos); ax.set_xticklabels(['β  (mean R)','β  (mean G)','β  (mean B)'])
ax.set_ylabel('fitted coefficient (standardised)')
ax.legend([bp1['boxes'][0],bp2['boxes'][0]],['Least squares (OLS)','Ridge  (λ = 10)'],frameon=False,fontsize=9,loc='upper right')
ax.set_title('300 refits, 30 leaves each — spread = instability',loc='left',fontsize=10.5,color=MUTE)
plt.tight_layout(); plt.savefig('fig/u2_ridge.png',dpi=200); plt.close()
print('sd OLS',co.std(0).round(3),'sd ridge',cr.std(0).round(3))
# ---- leaf cutouts (transparent) + photo strip
for k in range(5):
    for n,p in enumerate(pick[k][:2]):
        im=rgb(p); m=mask(im)
        m=cv2.GaussianBlur(m,(5,5),0)
        rgba=np.dstack([im,m]); cv2.imwrite(f'fig/leaf_{k}_{n}.png',cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(f'fig/photo_{k}_{n}.jpg',cv2.cvtColor(im,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,92])
# pipeline example: original, mask, segmented, gray texture
im=rgb(pick[1][0]); m=mask(im)
seg=im.copy(); seg[m==0]=255
gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY); gray[m==0]=255
cv2.imwrite('fig/pipe_0.jpg',cv2.cvtColor(im,cv2.COLOR_RGB2BGR)); cv2.imwrite('fig/pipe_1.png',m)
cv2.imwrite('fig/pipe_2.jpg',cv2.cvtColor(seg,cv2.COLOR_RGB2BGR)); cv2.imwrite('fig/pipe_3.jpg',gray)
print(pick[1][0])

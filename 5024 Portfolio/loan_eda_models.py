import os, warnings, json
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
os.makedirs('figures', exist_ok=True)
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import *
sns.set_theme(style='whitegrid',font='DejaVu Sans')
d=pd.read_csv('loan_data.csv'); X=['fico','int.rate','dti','revol.util']; y='not.fully.paid'
D=d[X+[y]]; R={}
R['n']=len(D); R['miss']=int(D.isna().sum().sum()); R['dups_full']=int(d.duplicated().sum()); R['dups_sel']=int(D.duplicated().sum())
R['desc']=D[X].describe().T.round(3).to_dict('index')
R['skew']=D[X].skew().round(3).to_dict()
out={}
for c in X:
    q1,q3=D[c].quantile([.25,.75]); i=q3-q1; lo,hi=q1-1.5*i,q3+1.5*i
    out[c]=dict(lo=round(lo,3),hi=round(hi,3),n=int(((D[c]<lo)|(D[c]>hi)).sum()))
R['outliers']=out; R['revol_gt100']=int((D['revol.util']>100).sum())
R['by_class']=D.groupby(y)[X].mean().round(3).to_dict('index'); R['by_class_med']=D.groupby(y)[X].median().round(3).to_dict('index')
tests={}
for c in X:
    a=D[D[y]==0][c]; b=D[D[y]==1][c]; u,p=stats.mannwhitneyu(a,b); 
    tests[c]=dict(U_p=float(p), pb=round(stats.pointbiserialr(D[y],D[c])[0],3))
R['tests']=tests
R['corr']=D[X].corr().round(3).to_dict(); R['corr_y']=D.corr()[y].round(3).to_dict()
# figures
fig,ax=plt.subplots(1,2,figsize=(8,3.2)); 
D[y].map({0:'Paid',1:'Not_Paid'}).value_counts().reindex(['Paid','Not_Paid']).plot.bar(ax=ax[0],color=['#2E75B6','#C55A11']); ax[0].set_title('Class distribution'); ax[0].set_xlabel(''); ax[0].tick_params(axis='x',rotation=0)
for p in ax[0].patches: ax[0].annotate(int(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),ha='center',va='bottom',fontsize=9)
ax[1].axis('off'); plt.tight_layout(); fig.delaxes(ax[1]); plt.savefig('figures/class.png',dpi=160); plt.show(); plt.close()
fig,ax=plt.subplots(2,2,figsize=(8,6))
for a,c in zip(ax.ravel(),X): sns.histplot(D[c],bins=30,kde=True,ax=a,color='#2E75B6'); a.set_title(f'Distribution of {c}')
plt.tight_layout(); plt.savefig('figures/hist.png',dpi=160); plt.show(); plt.close()
fig,ax=plt.subplots(1,4,figsize=(10,3.4)); L=D.assign(Outcome=D[y].map({0:'Paid',1:'Not_Paid'}))
for a,c in zip(ax,X): sns.boxplot(data=L,x='Outcome',y=c,ax=a,order=['Paid','Not_Paid'],palette=['#2E75B6','#C55A11'],hue='Outcome',legend=False,fliersize=2); a.set_title(c); a.set_xlabel('')
plt.tight_layout(); plt.savefig('figures/box.png',dpi=160); plt.show(); plt.close()
plt.figure(figsize=(5.2,4.2)); sns.heatmap(D[X+[y]].corr(),annot=True,fmt='.2f',cmap='RdBu_r',center=0,vmin=-1,vmax=1); plt.title('Correlation matrix'); plt.tight_layout(); plt.savefig('figures/corr.png',dpi=160); plt.show(); plt.close()
g=sns.pairplot(L.sample(1500,random_state=1),vars=['fico','int.rate'],hue='Outcome',palette=['#2E75B6','#C55A11'],plot_kws=dict(s=10,alpha=.5),height=2.6); g.savefig('figures/pair.png',dpi=160); plt.show(); plt.close('all')
# models
Xtr,Xte,ytr,yte=train_test_split(D[X],D[y],test_size=.2,stratify=D[y],random_state=42)
M={'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=1000)),
   'Naive Bayes':GaussianNB(),
   'Decision Tree':DecisionTreeClassifier(max_depth=4,min_samples_leaf=50,class_weight='balanced',random_state=42),
   'Random Forest':RandomForestClassifier(n_estimators=300,min_samples_leaf=20,class_weight='balanced',random_state=42,n_jobs=-1)}
res={}; cv=StratifiedKFold(5,shuffle=True,random_state=42); rocs={}
for n,m in M.items():
    c=cross_validate(m,Xtr,ytr,cv=cv,scoring=['roc_auc','recall','f1'])
    m.fit(Xtr,ytr); p=m.predict(Xte); pr=m.predict_proba(Xte)[:,1]
    tn,fp,fn,tp=confusion_matrix(yte,p).ravel()
    res[n]=dict(acc=accuracy_score(yte,p),prec=precision_score(yte,p,zero_division=0),rec=recall_score(yte,p),f1=f1_score(yte,p),auc=roc_auc_score(yte,pr),cv_auc=c['test_roc_auc'].mean(),cv_rec=c['test_recall'].mean(),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
    rocs[n]=roc_curve(yte,pr)
res['Majority baseline']=dict(acc=1-yte.mean(),prec=0,rec=0,f1=0,auc=.5,cv_auc=.5,cv_rec=0)
R['models']={k:{a:round(b,3) if isinstance(b,float) else b for a,b in v.items()} for k,v in res.items()}
R['test_n']=len(yte); R['test_pos']=int(yte.sum())
lr=M['Logistic Regression'][-1]; R['lr_coef']=dict(zip(X,lr.coef_[0].round(3)))
R['rf_imp']=dict(zip(X,M['Random Forest'].feature_importances_.round(3)))
plt.figure(figsize=(5.2,4.2))
for n,(f,t,_) in rocs.items(): plt.plot(f,t,label=f"{n} (AUC {res[n]['auc']:.2f})")
plt.plot([0,1],[0,1],'k--',lw=1); plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title('ROC curves (test set)'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig('figures/roc.png',dpi=160); plt.show(); plt.close()
fig,ax=plt.subplots(1,2,figsize=(8,3.2))
for a,n in zip(ax,['Logistic Regression','Random Forest']):
    cm=np.array([[res[n]['tn'],res[n]['fp']],[res[n]['fn'],res[n]['tp']]]); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=False,ax=a,xticklabels=['Paid','Not_Paid'],yticklabels=['Paid','Not_Paid']); a.set_title(n); a.set_xlabel('Predicted'); a.set_ylabel('Actual')
plt.tight_layout(); plt.savefig('figures/cm.png',dpi=160); plt.show(); plt.close()
json.dump(R,open('R.json','w'),indent=1,default=float)
print(json.dumps(R,indent=1,default=float))

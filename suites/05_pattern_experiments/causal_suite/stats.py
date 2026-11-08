import numpy as np
from scipy.stats import norm as normal
from sklearn.metrics import roc_auc_score


def holm(ps):
 ps=np.asarray(ps,float);out=np.full(len(ps),np.nan);valid=np.flatnonzero(np.isfinite(ps));order=valid[np.argsort(ps[valid])]
 running=0.
 for rank,i in enumerate(order):running=max(running,min(1.,(len(order)-rank)*ps[i]));out[i]=running
 return out

def boot(values,groups=None,n=2000,seed=0,median=False):
 v=np.asarray(values,float);g=np.asarray(groups if groups is not None else np.arange(len(v)));ok=np.isfinite(v);v=v[ok];g=g[ok]
 if not len(v):return dict(estimate=None,lo=None,hi=None,n=0,groups=0)
 unique=np.unique(g);blocks=[v[g==u] for u in unique];fn=np.median if median else np.mean;rng=np.random.default_rng(seed)
 samples=[fn(np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])) for _ in range(n)] if len(unique)>1 else []
 return dict(estimate=float(fn(v)),lo=float(np.quantile(samples,.025)) if samples else None,hi=float(np.quantile(samples,.975)) if samples else None,n=len(v),groups=len(unique))

def paired_p(values,groups,n=2000,seed=0):
 v=np.asarray(values,float);g=np.asarray(groups);ok=np.isfinite(v);v=v[ok];g=g[ok]
 if len(set(g))<2:return None
 sums=np.array([v[g==u].sum() for u in np.unique(g)]);rng=np.random.default_rng(seed);observed=abs(sums.sum());count=0
 for _ in range(n):count+=abs((sums*rng.choice([-1,1],len(sums))).sum())>=observed-1e-12
 return float((count+1)/(n+1))

def auc_test(y,x,groups,n=2000,seed=0):
 y=np.asarray(y);x=np.asarray(x,float);g=np.asarray(groups);ok=np.isfinite(x);x=x[ok];y=y[ok];g=g[ok]
 if len(np.unique(y))<2:return dict(auc=None,lo=None,hi=None,p=None,n=len(y))
 # Permute group labels only if each independent subject has one class.
 unique=np.unique(g);blocks=[np.flatnonzero(g==u) for u in unique];rng=np.random.default_rng(seed)
 obs=roc_auc_score(y,x);bs=[]
 for _ in range(n):
  ix=np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])
  if len(np.unique(y[ix]))==2:bs.append(roc_auc_score(y[ix],x[ix]))
 p=None
 if all(len(set(y[b]))==1 for b in blocks):
  labels=np.array([y[b[0]] for b in blocks]);count=0
  for _ in range(n):
   perm=rng.permutation(labels);yp=y.copy()
   for b,label in zip(blocks,perm):yp[b]=label
   count+=abs(roc_auc_score(yp,x)-.5)>=abs(obs-.5)
  p=(count+1)/(n+1)
 return dict(auc=float(obs),lo=float(np.quantile(bs,.025)) if bs else None,hi=float(np.quantile(bs,.975)) if bs else None,p=p,n=len(y))

def trend_test(successes,totals,steps):
 """Requested Cochran-Armitage statistic. Descriptive p assumes independence;
 longitudinal prompt-cluster bootstrap is reported separately by report.py.
 """
 a=np.asarray(successes,float);n=np.asarray(totals,float);w=np.log(np.asarray(steps,float));keep=n>0;a=a[keep];n=n[keep];w=w[keep]
 if len(n)<2:return dict(z=None,p_naive=None)
 p=a.sum()/n.sum();mean=np.average(w,weights=n);var=p*(1-p)*np.sum(n*(w-mean)**2)
 if var<=0:return dict(z=None,p_naive=None)
 z=np.sum(w*(a-n*p))/np.sqrt(var);return dict(z=float(z),p_naive=float(2*normal.sf(abs(z))))


def fixed_effects(values,models,groups):
 """OLS sum-coded model FE; intercept = equal-model mean paired difference.
 Subject-cluster sandwich covariance with finite-sample correction.
 """
 from scipy.stats import t
 y=np.asarray(values,float);models=np.asarray(models);groups=np.asarray(groups);levels=np.unique(models);G=len(np.unique(groups))
 X=np.ones((len(y),len(levels)))
 for i,level in enumerate(levels[:-1]):X[:,i+1]=(models==level).astype(float)-(models==levels[-1]).astype(float)
 if G<2 or len(y)<=X.shape[1]:return dict(estimate=None,lo=None,hi=None,p=None)
 inv=np.linalg.pinv(X.T@X);beta=inv@X.T@y;res=y-X@beta;meat=np.zeros_like(inv)
 for group in np.unique(groups):
  index=groups==group;score=X[index].T@res[index];meat+=np.outer(score,score)
 covariance=inv@meat@inv*(G/(G-1))*((len(y)-1)/(len(y)-X.shape[1]));se=np.sqrt(max(0,covariance[0,0]));critical=t.ppf(.975,G-1)
 return dict(estimate=float(beta[0]),lo=float(beta[0]-critical*se),hi=float(beta[0]+critical*se),p=float(2*t.sf(abs(beta[0]/se),G-1)) if se>0 else (1. if abs(beta[0])<1e-12 else 0.))

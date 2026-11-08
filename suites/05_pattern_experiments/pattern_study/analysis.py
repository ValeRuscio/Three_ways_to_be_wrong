from pathlib import Path
import numpy as np
from .common import *
from gate_study.variance import run as variance


def cos_matrix(X):
 n=np.linalg.norm(X,axis=1);valid=n>1e-12
 Y=X/np.maximum(n[:,None],1e-12)
 return Y@Y.T,valid


def dimension(X):
 if len(X)<2:return None
 X=X-X.mean(0);G=X@X.T;tr=np.trace(G);sq=(G*G).sum()
 return float(tr*tr/sq) if sq>1e-20 else None


def factor_summary(f,s,eps):
 # [neurons,3]: gate, subject-origin up, gold-readout; keep sign separately.
 cf=np.prod(f,axis=1);cs=np.prod(s,axis=1);valid=(np.abs(f)>eps).all(1)&(np.abs(s)>eps).all(1)
 total=np.abs(cs).sum();weight=np.abs(cs[valid]);retained=weight.sum()
 out=dict(neurons=len(f),valid_neurons=int(valid.sum()),excluded_neurons=int((~valid).sum()),retained_donor_absolute_contribution=float(retained/total) if total>0 else None,
  sign_flip_fraction=float(np.mean(np.sign(cf)!=np.sign(cs))),failure_contribution=float(cf.sum()),donor_contribution=float(cs.sum()))
 if not valid.any() or retained<=0:return out
 logs=np.log(np.abs(f[valid]))-np.log(np.abs(s[valid]));net=np.log(np.abs(cf[valid]))-np.log(np.abs(cs[valid]));w=weight/retained
 out['log_identity_max_error']=float(np.max(np.abs(logs.sum(1)-net)))
 for j,name in enumerate(['gate','up_subject','readout']):
  out[name+'_signed_log_change']=float(w@logs[:,j]);out[name+'_absolute_log_change']=float(w@np.abs(logs[:,j]))
 out['net_log_change']=float(w@net)
 den=sum(out[n+'_absolute_log_change'] for n in ['gate','up_subject','readout'])
 for n in ['gate','up_subject','readout']:out[n+'_absolute_change_share']=out[n+'_absolute_log_change']/den if den>1e-20 else None
 return out


def diagnose(root,cfg):
 metas=measured(root);ref={};rows=[]
 for rel in sorted({m['relation'] for m in metas}):
  grouped={};shares={}
  for m in metas:
   if m['relation']==rel and m['split']=='calibration' and m['correct']:
    grouped.setdefault(m['group_id'],[]).append(m['origins']['subject'])
    denominator=sum(abs(v) for v in m['origins'].values())
    if denominator>1e-20:shares.setdefault(m['group_id'],[]).append(m['origins']['subject']/denominator)
  vals=[np.mean(v) for v in grouped.values()]
  share_values=[np.mean(v) for v in shares.values()]
  ref[rel]=dict(subjects=len(vals),threshold=float(np.quantile(vals,cfg['source_quantile'])) if len(vals)>=3 else None,share_threshold=float(np.quantile(share_values,cfg['source_quantile'])) if len(share_values)>=3 else None)
 for m in metas:
  o=m['origins'];den=sum(abs(v) for v in o.values());threshold=ref[m['relation']]['threshold']
  label='correct' if m['correct'] else 'uncalibrated' if threshold is None else 'source_below_success_reference' if o['subject']<threshold else 'non_subject_counteraction' if sum(v for k,v in o.items() if k!='subject')<0 else 'other_failure'
  rows.append(dict(id=m['id'],group_id=m['group_id'],relation=m['relation'],split=m['split'],template=m['template'],correct=m['correct'],margin=m['margin'],label=label,
   source_success_threshold=threshold,share_success_threshold=ref[m['relation']]['share_threshold'],share_below_success_reference=(o['subject']/den<ref[m['relation']]['share_threshold']) if den>1e-20 and ref[m['relation']]['share_threshold'] is not None else None,subject_signed_absolute_share=o['subject']/den if den>1e-20 else None,**{f'origin_{k}':v for k,v in o.items()}))
 dump(Path(root)/'reports/source_calibration.json',ref);table(rows,Path(root)/'reports/origins_and_diagnostics.csv')


def run(root,cfg):
 root=Path(root);out=root/'reports';out.mkdir(exist_ok=True);metas=measured(root)
 ev=[m for m in metas if m['split']=='evaluation'];records=list({m['id']:m for m in ev}.values())
 variance(records,root);diagnose(root,cfg)
 table([{k:m.get(k) for k in ['id','group_id','relation','split','template','correct','full_answer_correct','margin','certificate','factor_certificate']} for m in metas],out/'coverage.csv')
 similarities=[];dimensions=[];distances=[];factorrows=[];pairdist=[]
 for rel in sorted({m['relation'] for m in ev}):
  cal=[m for m in metas if m['relation']==rel and m['split']=='calibration' and m['correct']]
  ca={}
  for m in cal:ca.setdefault(m['group_id'],[]).append(load_arrays(cache_path(root,m,m['template']))['gate'])
  centroid=np.stack([np.stack(v).mean(0) for v in ca.values()]).mean(0) if ca else None
  for t in sorted({m['template'] for m in ev}):
   ms=[m for m in ev if m['relation']==rel and m['template']==t]
   if not ms:continue
   X=np.stack([load_arrays(cache_path(root,m,t))['gate'] for m in ms]);labels=np.array([m['correct'] for m in ms]);groups=np.array([m['group_id'] for m in ms]);distinct=groups[:,None]!=groups[None,:]
   for l in range(X.shape[1]):
    for si,site in enumerate(['subject','answer']):
     vectors=X[:,l,si].astype(float);cos,valid=cos_matrix(vectors);maskbase=distinct&valid[:,None]&valid[None,:]&np.triu(np.ones_like(cos,dtype=bool),1)
     for name,mask in [('success_success',labels[:,None]&labels[None,:]),('failure_failure',~labels[:,None]&~labels[None,:]),('success_failure',labels[:,None]!=labels[None,:])]:
      values=cos[mask&maskbase];similarities.append(dict(relation=rel,template=t,layer=l,site=site,comparison=name,pairs=len(values),mean_cosine=float(values.mean()) if len(values) else None))
     for name,sel in [('success',labels),('failure',~labels),('all',np.ones(len(ms),bool))]:
      dimensions.append(dict(relation=rel,template=t,layer=l,site=site,condition=name,n=int(sel.sum()),participation_ratio=dimension(vectors[sel])))
     if centroid is not None:
      c=centroid[l,si].astype(float);cn=np.linalg.norm(c)
      for i,m in enumerate(ms):
       norm=np.linalg.norm(vectors[i]);distance=1-float(vectors[i]@c/(norm*cn)) if norm*cn>1e-12 else None
       distances.append(dict(id=m['id'],group_id=m['group_id'],relation=rel,template=t,layer=l,site=site,correct=m['correct'],margin=m['margin'],centroid_subjects=len(ca),cosine_distance=distance,euclidean_distance=float(np.linalg.norm(vectors[i]-c)),**{f'origin_{k}':v for k,v in m['origins'].items()}))
 for pair in pairs(root):
  m,d=pair['recipient'],pair['donor'];f=load_arrays(cache_path(root,m,m['template']));s=load_arrays(cache_path(root,d,d['template']))
  for l in range(len(f['gate'])):
   for si,site in enumerate(['subject','answer']):
    base=dict(pair_id=pair['pair_id'],id=m['id'],group_id=m['group_id'],relation=m['relation'],recipient_template=m['template'],donor_template=d['template'],layer=l,site=site)
    if m['g']==d['g']:
     F=np.stack([f['gate'][l,si],f['up_subject'][l,si],f['down_gold'][l,si]],axis=1).astype(float)
     S=np.stack([s['gate'][l,si],s['up_subject'][l,si],s['down_gold'][l,si]],axis=1).astype(float)
     factorrows.append(dict(**base,**factor_summary(F,S,cfg['log_ratio_epsilon'])))
    x=f['gate'][l,si].astype(float);y=s['gate'][l,si].astype(float);den=np.linalg.norm(x)*np.linalg.norm(y)
    pairdist.append(dict(**base,cosine_distance=1-float(x@y/den) if den>1e-12 else None,margin_gap=d['margin']-m['margin'],failure_margin=m['margin'],donor_margin=d['margin']))
 table(similarities,out/'pairwise_cosines.csv');table(dimensions,out/'effective_dimensions.csv');table(distances,out/'success_centroid_distances.csv');table(factorrows,out/'factor_log_ratios.csv');table(pairdist,out/'same_fact_template_distances.csv')
 dump(out/'interpretation.json',dict(
  rank='One input/target vector is rank one. Higher ranks use independent calibration pairs plus the target pair; calibration-only transfer is separate. Ridge LS followed by SVD truncation is reported honestly.',
  variance='Three variants t1,t2,t3; t3 prefixes t1 with Question:. Early variance includes token-identity effects. No unique causal handoff inferred.',
  geometry='Pairwise entries are dependent; do not use pair count as independent sample size. Centroids use calibration successes, averaged within subject. PCA participation ratio is sample-size dependent.',
  factors='Signed log-magnitude ratios; zeros excluded with retained contribution mass reported; sign flips separate. Absolute-change shares are descriptive, not causal percentages.',
  diagnostics='New contrastive, shift-invariant diagnostic labels; not the original taxonomy and not proved causal stages. Success-quantile calibration uses independent subjects.',
  curves='Half effect uses observed signed endpoint change, not monotonicity. Gate sign interventions can produce values outside native SiLU range; no discrete-program claim follows automatically.',
  training='Retrospective cohort selected on OLMo final successes. Acquisition times interval-censored; no promise of hundreds of events.',
  tokenizers='Every model uses its native tokenizer; fingerprints and common fact coverage are reported. Qwen is not same-tokenizer replication.'))

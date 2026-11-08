from pathlib import Path
import json
import numpy as np
from .common import cache_path,meta_path,read_meta,load_arrays,npz,dump,key,table
from .factors import SITES


def build(records,root,cfg):
 root=Path(root);out=root/'selectors';out.mkdir(parents=True,exist_ok=True);index={}
 for relation in sorted({r['relation'] for r in records}):
  # Average templates within subject before averaging across independent subjects.
  subjects={}
  for r in records:
   if r['relation']!=relation or r['split']!='calibration':continue
   good=[]
   for t in range(4):
    if meta_path(root,r,t).exists():
     m=read_meta(root,r,t)
     if m.get('status')=='measured' and m.get('correct'):good.append(cache_path(root,r,t))
   if good:subjects.setdefault(str(r['group_id']),[]).extend(good)
  n=len(subjects)
  if n<cfg['minimum_reference_subjects']:
   index[relation]=dict(status='insufficient_calibration_successes',subjects=n);continue
  sums={};squares={}
  for paths in subjects.values():
   vals={}
   for p in paths:
    a=load_arrays(p)
    for k in ['contribution_gold','contribution_margin','pre','gate','up_native','up_subject','down_gold','head_margin']:
     vals.setdefault(k,[]).append(a[k])
   for k,vs in vals.items():
    v=np.mean(vs,axis=0,dtype=np.float64);sums[k]=sums.get(k,0)+v;squares[k]=squares.get(k,0)+v*v
  mean={k:(v/n).astype(np.float32) for k,v in sums.items()}
  for k,v in sums.items():mean[k+'_std']=np.sqrt(np.maximum(0,(squares[k]-v*v/n)/max(n-1,1))).astype(np.float32)
  filename=relation.replace(' ','_')+'.npz';npz(out/filename,**mean)
  index[relation]=dict(status='ready',subjects=n,file=filename,reference_groups=sorted(subjects))
 dump(out/'index.json',index);return index


def positive_top(score,k):
 order=np.argsort(score)[::-1];return order[score[order]>0][:k]


def select_site(a,site,k):
 s=SITES.index(site);scores=a['contribution_gold'][:,s];ranked=[]
 for l,row in enumerate(scores):
  neurons=positive_top(row,k)
  if len(neurons):ranked.append((float(row[neurons].sum()),l,neurons))
 if not ranked:return None
 _,l,neurons=max(ranked,key=lambda x:x[0]);return dict(layer=l,site=site,site_index=s,neurons=neurons.tolist(),reference_pre=a['pre'][l,s,neurons].tolist())


def sets_for(r,root,index,cfg):
 root=Path(root);result=[];entry=index.get(r['relation'],{})
 if entry.get('status')=='ready':
  a=load_arrays(root/'selectors'/entry['file']);result.append(('independent_relation',a,entry['subjects']))
 # First successful alternative of the SAME fact; don't pool with independent sets.
 for t in [1,2,3]:
  if meta_path(root,r,t).exists():
   m=read_meta(root,r,t)
   if m.get('status')=='measured' and m.get('correct'):
    result.append((f'same_fact_donor_t{t}',load_arrays(cache_path(root,r,t)),1));break
 return result


def profiles(records,root,index,cfg):
 root=Path(root);rows=[];coverage=[]
 for r in records:
  if not cache_path(root,r,0).exists():continue
  m=read_meta(root,r,0);a=load_arrays(cache_path(root,r,0));sets=sets_for(r,root,index,cfg)
  coverage.append(dict(id=r['id'],relation=r['relation'],label=m['label'],sets=len(sets),correct=m['correct']))
  detail=[]
  for kind,ref,nref in sets:
   for site in cfg['sites']:
    s=SITES.index(site)
    for l in range(a['gate'].shape[0]):
     for K in sorted(set(cfg['profile_k']+[cfg['top_k']])):
      neurons=positive_top(ref['contribution_gold'][l,s],K)
      if not len(neurons):continue
      row=dict(id=str(r['id']),group_id=str(r['group_id']),relation=r['relation'],label=m['label'],correct=m['correct'],set_kind=kind,reference_subjects=nref,site=site,layer=l,k_requested=K,k_selected=len(neurons),margin=m['margin'],target_rank=m['target_rank'],template_margin=m['origin_margin']['template'])
      for f in ['pre','gate','sigmoid','up_native','up_subject','down_gold','down_competitor','contribution_gold','contribution_margin','up_cosine','down_cosine','signed_distance']:
       row[f+'_mean']=float(a[f][l,s,neurons].mean())
       if f in ref:
        diff=a[f][l,s,neurons]-ref[f][l,s,neurons];row[f+'_reference_difference']=float(diff.mean())
        if f+'_std' in ref:
         sd=ref[f+'_std'][l,s,neurons];valid=sd>1e-6;row[f+'_reference_z']=float((diff[valid]/sd[valid]).mean()) if valid.any() else None
      row['lower_pre_than_reference_fraction']=float((a['pre'][l,s,neurons]<ref['pre'][l,s,neurons]).mean())
      for level in cfg['preactivation_levels']:
       distance=(a['pre'][l,s,neurons]-level)/np.maximum(a['gate_row_norm'][l,s,neurons],1e-20)
       row[f'level_{level}_signed_distance_mean']=float(distance.mean())
      competitor=positive_top(a['gate'][l,s]*a['up_subject'][l,s]*a['down_competitor'][l,s],K)
      row['competitor_set_overlap']=len(set(neurons)&set(competitor))/len(neurons)
      rows.append(row)
      if K==cfg['top_k']:
       for neuron in neurons:
        detail.append(dict(set_kind=kind,site=site,layer=l,neuron=int(neuron),pre=float(a['pre'][l,s,neuron]),reference_pre=float(ref['pre'][l,s,neuron]),
         gate=float(a['gate'][l,s,neuron]),up_subject=float(a['up_subject'][l,s,neuron]),down_gold=float(a['down_gold'][l,s,neuron]),down_competitor=float(a['down_competitor'][l,s,neuron]),signed_distance=float(a['signed_distance'][l,s,neuron])))
  dump(root/'profiles'/f'{key(r)}.json',dict(id=str(r['id']),rows=detail))
 table(rows,root/'reports/deficit_profiles.csv');table(coverage,root/'reports/neuron_set_coverage.csv')

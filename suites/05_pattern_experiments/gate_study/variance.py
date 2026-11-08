"""Balanced crossed subject x template decomposition, with interaction retained.
No zero-padding of attention rows. Attention role masses are exact coarsenings.
"""
import numpy as np
from pathlib import Path
from .common import load_arrays,cache_path,meta_path,read_meta,npz,table,dump


def crossed(X):
 X=np.asarray(X,dtype=np.float64)
 if X.ndim<3 or X.shape[0]<2 or X.shape[1]<2:raise ValueError('Need >=2 subjects and >=2 templates')
 n,k=X.shape[:2];grand=X.mean((0,1));sm=X.mean(1);tm=X.mean(0)
 subject=k*((sm-grand)**2).sum(0);template=n*((tm-grand)**2).sum(0)
 residual=X-sm[:,None]-tm[None]+grand;interaction=(residual**2).sum((0,1));total=((X-grand)**2).sum((0,1))
 error=float(np.max(np.abs(total-subject-template-interaction))/max(1,float(total.max())))
 if error>1e-10:raise AssertionError('ANOVA sum-of-squares identity failed')
 return dict(subject=subject,template=template,interaction=interaction,total=total,certificate=error)


def run(records,root):
 root=Path(root);out=root/'reports';rows=[];coverage=[];raw=[];heads=[]
 for relation in sorted({r['relation'] for r in records}):
  # One observation per subject in a relation, selected by ID without outcomes.
  unique={}
  for r in sorted(records,key=lambda r:str(r['id'])):
   if r['relation']==relation:unique.setdefault(str(r['group_id']),r)
  valid=[]
  for r in unique.values():
   if all(meta_path(root,r,t).exists() and read_meta(root,r,t).get('status')=='measured' and cache_path(root,r,t).exists() for t in [1,2,3]):valid.append(r)
  coverage.append(dict(relation=relation,subjects_available=len(unique),complete_subjects=len(valid),templates=3))
  if len(valid)<2:continue
  # Load only needed arrays. Attention compares four semantic masses, not raw slots.
  data=[]
  for r in valid:
   triple=[]
   for t in [1,2,3]:
    with np.load(cache_path(root,r,t),allow_pickle=False) as z:triple.append(dict(gate=z['gate'],attention_roles=z['attention_roles']))
   data.append(triple)
  for name in ['subject_gate','answer_gate','attention_roles']:
   X=np.array([[d['gate'][:,0 if name=='subject_gate' else 1] if name!='attention_roles' else d['attention_roles'] for d in triple] for triple in data])
   a=crossed(X);npz(out/'variance_coordinates'/f'{relation.replace(" ","_")}_{name}.npz',**a)
   if name=='attention_roles':
    for l in range(X.shape[2]):
     for h in range(X.shape[3]):
      total=float(a['total'][l,h].sum());record=dict(relation=relation,layer=l,head=h,subjects=len(valid),total_ss=total)
      for component in ['subject','template','interaction']:
       ss=float(a[component][l,h].sum());record[component+'_ss']=ss;record[component+'_fraction']=ss/total if total>1e-20 else None
      heads.append(record)
   for layer in range(X.shape[2]):
    total=float(a['total'][layer].sum())
    row=dict(relation=relation,measurement=name,layer=layer,subjects=len(valid),templates=3,total_ss=total,certificate=a['certificate'])
    for component in ['subject','template','interaction']:
     ss=float(a[component][layer].sum());row[component+'_ss']=ss;row[component+'_fraction']=ss/total if total>1e-20 else None
    rows.append(row)
  # Uncoarsened rows: between-subject variation ONLY within an exact layout.
  # Across-template raw-row decomposition is undefined without a coordinate map.
  for t in [1,2,3]:
   strata={}
   for r in valid:
    m=read_meta(root,r,t);S=set(m['source']);layout=tuple(None if i in S else v for i,v in enumerate(m['input_ids']));strata.setdefault(layout,[]).append(r)
   for index,rs in enumerate(strata.values()):
    if len(rs)<2:continue
    arrays=[]
    for r in rs:
     with np.load(cache_path(root,r,t),allow_pickle=False) as z:arrays.append(z['attention_rows'].astype(float))
    A=np.stack(arrays);v=((A-A.mean(0))**2).mean(0).sum(-1)
    for l in range(v.shape[0]):
     for h in range(v.shape[1]):raw.append(dict(relation=relation,template=t,stratum=index,subjects=len(rs),layer=l,head=h,row_squared_variation=v[l,h]))
 table(heads,out/'variance_head_components.csv')
 table(rows,out/'variance_components.csv');table(coverage,out/'variance_coverage.csv');table(raw,out/'attention_exact_layout_variation.csv')
 crossings=[]
 for name in ['subject_gate','answer_gate','attention_roles']:
  rr=[r for r in rows if r['measurement']==name];layers=sorted({r['layer'] for r in rr});last=None
  for l in layers:
   s=sum(r['subject_ss'] for r in rr if r['layer']==l);t=sum(r['template_ss'] for r in rr if r['layer']==l);diff=s-t
   if last is not None and diff*last[1]<0:crossings.append(dict(measurement=name,between_layers=[last[0],l]))
   last=(l,diff)
 dump(out/'variance_interpretation.json',dict(crossings=crossings,
  interpretation='Exact descriptive ANOVA over the complete subject x three-paraphrase grid. Interaction is not noise. Gates at last-subject / final prompt token. Attention is four role masses; raw rows have separate within-layout variation. Crossings are descriptive, not established routing-to-content hand-offs.'))

"""Success-calibrated, vocabulary-centered origin diagnostics, NOT causal stages.
Keep continuation failures and unresolved cases instead of forcing three labels.
"""
import numpy as np


def calibrate(records,quantile=.05,min_subjects=5):
 out={}
 for relation in sorted({r['prompt']['relation'] for r in records}):
  by={}
  for r in records:
   if r['prompt']['relation']!=relation or not r['answer_correct'] or not r['canonical_top1']:continue
   o=r['origins'];share=o['gold_source_share'];tau=o['centered_gold']['subject']
   if share is None:continue
   by.setdefault(r['prompt']['group_id'],[]).append([tau,share])
  values=np.array([np.mean(v,axis=0) for v in by.values()])
  out[relation]=dict(subjects=len(values),tau=float(np.quantile(values[:,0],quantile)) if len(values)>=min_subjects else None,
   share=float(np.quantile(values[:,1],quantile)) if len(values)>=min_subjects else None,quantile=quantile)
 return out


def classify(record,reference):
 if record['answer_correct']:return 'answer_success'
 if record['canonical_top1']:return 'continuation_failure'
 ref=reference.get(record['prompt']['relation'],{})
 if ref.get('tau') is None:return 'uncalibrated_failure'
 o=record['origins'];gs=o['gold_source_share'];cs=o['competitor_source_share']
 if gs is None or cs is None:return 'undefined_share'
 gold=o['centered_gold']['subject']>max(0,ref['tau']) and gs>max(0,ref['share'])
 competitor=o['centered_competitor']['subject']>max(0,ref['tau']) and cs>max(0,ref['share'])
 margin=o['margin_origins'];non_source=sum(v for k,v in margin.items() if k!='subject')
 if competitor and margin['subject']<0:return 'misretrieved_origin'
 if gold and margin['subject']>0 and non_source<0:return 'overridden_origin'
 if not gold and not competitor:return 'absent_origin'
 return 'mixed_origin_failure'

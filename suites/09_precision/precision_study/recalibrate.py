"""Report-only exploratory recalibration. Original independent labels stay primary."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from channel_study.io import dump,key,load
from channel_study.report import summarize
from followup_study.regimes import classify


def folds(subjects,n=5,seed=2026):
 ordered=sorted(set(map(str,subjects)),key=lambda x:hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
 return {s:i%n for i,s in enumerate(ordered)}


def references(train,minimum,q):
 # Only successful canonical-answer prompts calibrate the canonical-gold origin.
 # Each subject contributes one mean, irrespective of number of paraphrases.
 good=train[train.answer_correct & train.canonical_top1]
 refs={}
 for relation in sorted(train.relation.unique()):
  x=good[good.relation==relation].groupby('group_id')[['gold_subject','gold_source_share']].mean().dropna()
  refs[relation]=dict(subjects=len(x),tau=float(x.gold_subject.quantile(q)) if len(x)>=minimum else None,
   share=float(x.gold_source_share.quantile(q)) if len(x)>=minimum else None,quantile=q)
 return refs


def label(row,refs):
 return classify(dict(answer_correct=bool(row.answer_correct),canonical_top1=bool(row.canonical_top1),prompt={'relation':row.relation},
  origins=dict(centered_gold={'subject':row.gold_subject},centered_competitor={'subject':row.competitor_subject},
   gold_source_share=None if pd.isna(row.gold_source_share) else row.gold_source_share,
   competitor_source_share=None if pd.isna(row.competitor_source_share) else row.competitor_source_share,
   margin_origins={k:getattr(row,'margin_'+k) for k in ['subject','template','first','bias']})),refs)


def raw_join(evidence,out,labels,raw_roots):
 if not raw_roots:
  return dict(status='not_requested',reason='Aggregate profile summaries cannot be relabelled. Raw per-prompt split.json files required.')
 cohort=load(evidence/'data/cohort.json');prompts={(str(r['id']),str(r['template'])):r for r in cohort['evaluation']}
 roots=[]
 for root in map(Path,raw_roots):
  roots.append(root);roots.extend(p for p in root.glob('0[1-5]*') if p.is_dir())
 # Pin checks prevent joining nominal model names across different checkpoints.
 sources={}
 for model in labels.model.unique():
  pinned=load(evidence/model/'model.json')
  for root in roots:
   mp=root/model/'model.json'
   if mp.exists():
    pin=load(mp)
    if pin['repo']==pinned['repo'] and pin['sha']==pinned['sha']:sources.setdefault(model,[]).append(root/model)
 rows=[];missing=0
 # Both minimum-support sensitivities at q=.05; keep their scheme explicit.
 chosen=labels[np.isclose(labels['quantile'],.05)]
 for r in chosen.itertuples():
  prompt=prompts.get((str(r.id),str(r.template)))
  paths=[p/'prompts'/key(prompt)/'split.json' for p in sources.get(r.model,[])] if prompt else []
  path=next((p for p in paths if p.exists()),None)
  if path is None:missing+=1;continue
  meta=dict(model=r.model,id=r.id,group_id=r.group_id,template=r.template,relation=r.relation,regime=r.crossfit_regime,minimum_subjects=r.minimum_subjects)
  rows.extend(dict(meta,**x) for x in load(path))
 if rows:
  df=summarize(pd.DataFrame(rows),['model','minimum_subjects','regime','layer','family'],['selection_fraction','content_selection_cosine','content_norm','selection_norm','cross'],dict(seed=0,bootstrap=1000))
  df.to_csv(out/'exploratory_crossfit_profiles.csv',index=False)
 return dict(status='partial' if missing else 'complete',missing_labelled_prompt_rows=missing,available_gradient_rows=len(rows))


def run(evidence,out,raw_roots=()):
 evidence=Path(evidence);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 dtype={'model':str,'id':str,'group_id':str,'template':str}
 orig=pd.read_csv(evidence/'reports/regimes.csv',dtype=dtype)
 cov=pd.read_csv(evidence/'reports/coverage.csv',dtype=dtype)
 keys=['model','id','group_id','relation','template']
 data=orig.merge(cov[keys+['answer_correct','canonical_top1']],on=keys,validate='one_to_one')
 if len(data)!=len(cov):raise ValueError('Missing origins for coverage records')
 for col in ['answer_correct','canonical_top1']:
  if data[col].dtype!=bool:raise ValueError('Expected explicit Boolean correctness columns')
 mapping=folds(data.group_id);data['fold']=data.group_id.map(mapping)
 pd.DataFrame([dict(group_id=k,fold=v) for k,v in mapping.items()]).to_csv(out/'subject_folds.csv',index=False)
 results=[];thresholds=[]
 for model,model_data in data.groupby('model'):
  for fold in range(5):
   train=model_data[model_data.fold!=fold];test=model_data[model_data.fold==fold]
   assert not set(train.group_id)&set(test.group_id)
   for minimum in [5,20]:
    for q in [.01,.05,.1]:
     refs=references(train,minimum,q)
     thresholds.extend(dict(model=model,fold=fold,minimum_subjects=minimum,relation=relation,**v) for relation,v in refs.items())
     for row in test.itertuples():
      results.append(dict(model=model,id=row.id,group_id=row.group_id,relation=row.relation,template=row.template,fold=fold,
       minimum_subjects=minimum,quantile=q,original_regime=row.regime,crossfit_regime=label(row,refs),answer_correct=row.answer_correct))
 labels=pd.DataFrame(results);labels.to_csv(out/'exploratory_crossfit_labels.csv',index=False)
 pd.DataFrame(thresholds).to_csv(out/'exploratory_thresholds.csv',index=False)
 summary=labels.groupby(['model','minimum_subjects','quantile','crossfit_regime']).agg(prompts=('id','size'),facts=('id','nunique'),subjects=('group_id','nunique')).reset_index()
 summary.to_csv(out/'exploratory_counts.csv',index=False)
 # Explicit denominators: all errors, not only label-eligible errors.
 rows=[]
 for model,d in data.groupby('model'):
  errors=d[~d.answer_correct];base=dict(model=model,incorrect_prompts=len(errors))
  rows.append(dict(base,scheme='original_independent',uncalibrated=int((errors.regime=='uncalibrated_failure').sum()),uncalibrated_fraction=float((errors.regime=='uncalibrated_failure').mean())))
  for minimum in [5,20]:
   x=labels[(labels.model==model)&(labels.minimum_subjects==minimum)&np.isclose(labels['quantile'],.05)&~labels.answer_correct]
   rows.append(dict(base,scheme=f'exploratory_crossfit_min{minimum}',uncalibrated=int((x.crossfit_regime=='uncalibrated_failure').sum()),uncalibrated_fraction=float((x.crossfit_regime=='uncalibrated_failure').mean())))
 pd.DataFrame(rows).to_csv(out/'calibration_coverage_comparison.csv',index=False)
 # Conservative alternative requested: retain original labels on larger models.
 restricted=orig[orig.model.isin(['llama8b','olmo7b'])].copy()
 restricted.to_csv(out/'original_8b_olmo_labels.csv',index=False)
 restricted.groupby(['model','relation','regime']).agg(prompts=('id','size'),subjects=('group_id','nunique')).reset_index().to_csv(out/'original_8b_olmo_counts.csv',index=False)
 dump(out/'status.json',dict(profile_join=raw_join(evidence,out,labels,raw_roots),
  primary='Original independent calibration, with coverage explicit; restricted 8B/OLMo option preserves uncalibrated cases.',
  exploratory='Five-fold subject-held-out calibration from evaluation successes across existing templates. Same subject excluded across all templates/relations. Not new independent calibration data.',
  limitations='No pooled cross-relation fallback and no forcing labels. Min5 is sparse sensitivity; min20 is still a small lower-tail reference. Labels do not establish causal stages. Fold-specific thresholds may vary. Original profile summaries cannot be relabelled without raw split records.',
  uncertainty='No inference CI for learned crossfit labels is provided: uncertainty would require resampling subjects and refitting thresholds. Counts are descriptive.'))
 print(pd.DataFrame(rows).to_string(index=False),flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--out',required=True);p.add_argument('--raw-profiles',nargs='*',default=[]);a=p.parse_args();run(a.evidence,a.out,a.raw_profiles)
if __name__=='__main__':main()

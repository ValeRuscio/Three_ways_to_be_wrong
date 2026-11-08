import json
from pathlib import Path
from causal_suite.data import build_fixed,TEMPLATES,digest
from channel_study.fresh import render_cohort
from channel_study.io import dump,load,key


def prepare(cfg,out,previous=None):
 out=Path(out);source=out/'data';dest=source/'cohort.json'
 if dest.exists():return load(dest)
 previous=Path(previous) if previous else None
 rawpath=previous/'data_source/fixed_data.json' if previous else None
 if rawpath and rawpath.exists():raw=load(rawpath)
 else:raw=build_fixed(dict(seed=cfg['seed'],facts_jsonl=None,calibration_candidates_per_relation=100,evaluation_per_relation=100),source)
 if previous and (previous/'prompts.jsonl').exists():evaluation=[json.loads(s) for s in (previous/'prompts.jsonl').read_text().splitlines() if s.strip()]
 else:evaluation,_=render_cohort(raw)
 evaluation=[r for r in evaluation if r['template'] in cfg['templates']]
 subjects={str(r['group_id']) for r in evaluation};cal=[]
 for r in raw['rows']:
  if r['split']!='calibration' or str(r['group_id']) in subjects or not isinstance(r['subject'],str) or not r['subject'].strip():continue
  for t in cfg['calibration_templates']:cal.append(dict(r,template=t,prompt=TEMPLATES[r['relation']][t].format(s=r['subject'])))
 if not evaluation or not cal:raise ValueError('Need evaluation and independent calibration prompts')
 for r in evaluation+cal:r.update(id=str(r['id']),group_id=str(r['group_id']))
 if len({key(r) for r in evaluation})!=len(evaluation):raise ValueError('Duplicate evaluation prompt')
 result=dict(evaluation=evaluation,calibration=cal,dataset_revision=raw.get('revision'),previous=str(previous) if previous else None)
 dump(dest,result);return result


def selected(rows,count,seed):
 ordered=sorted(rows,key=lambda r:digest([seed,key(r)]))
 return {key(r) for r in (ordered if count is None else ordered[:count])}

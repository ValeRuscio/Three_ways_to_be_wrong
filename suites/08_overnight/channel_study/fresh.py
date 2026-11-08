"""Create outcome-independent evaluation prompts directly from PopQA."""
import argparse,json
from pathlib import Path
from causal_suite.data import build_fixed,TEMPLATES,digest
from .io import dump


def render_cohort(raw):
 rows=raw['rows'];cal=[r for r in rows if r['split']=='calibration' and isinstance(r['subject'],str) and r['subject'].strip()]
 prompts=[];excluded=[]
 for r in rows:
  if r['split']!='evaluation':continue
  if not isinstance(r['subject'],str) or not r['subject'].strip():
   excluded.append(dict(id=r['id'],reason='missing subject'));continue
  templates=TEMPLATES[r['relation']]
  texts=[t.format(s=r['subject']) for t in templates]
  texts.append('Question: '+templates[1].format(s=r['subject']))
  examples=[];seen=set()
  for q in sorted(cal,key=lambda q:digest([r['id'],q['id']])):
   if q['relation']!=r['relation'] or q['group_id']==r['group_id'] or q['group_id'] in seen:continue
   seen.add(q['group_id']);examples.append(templates[1].format(s=q['subject'])+' '+q['answer'])
   if len(examples)==3:break
  if len(examples)==3:texts.append('\n\n'.join(examples+[templates[1].format(s=r['subject'])]))
  else:excluded.append(dict(id=r['id'],template=4,reason='fewer than three calibration subjects'))
  for t,text in enumerate(texts):prompts.append(dict(r,template=t,prompt=text))
 return prompts,excluded


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=0)
 p.add_argument('--evaluation-per-relation',type=int,default=100);a=p.parse_args()
 root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
 if (root/'prompts.jsonl').exists():raise ValueError('prompts.jsonl already exists; use the saved cohort or a new output folder')
 spec=dict(seed=a.seed,facts_jsonl=None,calibration_candidates_per_relation=100,evaluation_per_relation=a.evaluation_per_relation)
 source=root/'data_source'
 if (source/'sampling.json').exists() and json.loads((source/'sampling.json').read_text())!=spec:raise ValueError('Sampling changed: use a new output folder')
 dump(source/'sampling.json',spec)
 try:raw=build_fixed(spec,source)
 except ModuleNotFoundError as e:
  if e.name=='datasets':raise RuntimeError('Fresh PopQA loading needs the datasets package in the selected existing Python environment. No prior results are needed.') from e
  raise
 prompts,excluded=render_cohort(raw)
 if not prompts:raise ValueError('No valid evaluation prompts')
 dest=root/'prompts.jsonl';tmp=dest.with_suffix('.jsonl.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in prompts));tmp.replace(dest)
 dump(source/'prompt_provenance.json',dict(source=raw['source'],revision=raw.get('revision'),sampling=spec,excluded=excluded,
  facts=len({r['id'] for r in prompts}),prompts=len(prompts),selection='Subject-disjoint hash split; no model-correctness filtering'))
 print(f'Created {len(prompts)} prompts / {len({r["id"] for r in prompts})} facts. No old results or cached activations used.',flush=True)
 print('Saved:',dest,flush=True)
if __name__=='__main__':main()

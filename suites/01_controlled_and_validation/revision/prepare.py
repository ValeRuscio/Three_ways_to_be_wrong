"""Convert legacy cohorts to an explicit, group-disjoint decision-point protocol.
Does not implement semantic divergence detection. Decoys require an independent pool.
"""
import argparse
import numpy as np
from revision.common import read_jsonl,manifest,digest,validate_records,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',required=True); p.add_argument('--out',required=True)
    p.add_argument('--task',required=True); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--calibration-fraction',type=float,default=.3)
    p.add_argument('--decision-scope',required=True,choices=['first_token','aligned_divergence'])
    p.add_argument('--allow-copy',action='store_true'); args=p.parse_args()
    if not 0<args.calibration_fraction<1: p.error('Calibration fraction must be between 0 and 1')
    out=manifest(args.out,args,[args.cohort]); records=read_jsonl(args.cohort); clean=[]; excluded=[]
    for i,r in enumerate(records):
        r=dict(r)
        if not args.allow_copy and r.get('copy_contaminated') and args.task!='extraction':
            excluded.append(dict(index=i,reason='copy_contaminated')); continue
        if not r['correct'] and r.get('competitor_token',-1)==-1:
            excluded.append(dict(index=i,reason='no_first_token_divergence')); continue
        group=r.get('group_id',r.get('subject'))
        if group is None: p.error('Each record needs group_id or subject; do not use row IDs as independent facts')
        if 'relation' not in r: p.error('Missing relation; restore it from source data rather than guessing')
        if args.decision_scope=='aligned_divergence' and 'input_ids' not in r:
            p.error('Aligned divergence requires explicit input_ids containing the shared generated prefix')
        r.update(id=str(r.get('id',digest([r['prompt'],r['target_first_token'],i])[:16])),
                 group_id=str(group),task=args.task,decision_scope=args.decision_scope)
        clean.append(r)
    eligible=sorted({r['group_id'] for r in clean if r['correct']})
    if len(eligible)<2: p.error('Need successful examples in at least two groups')
    rng=np.random.default_rng(args.seed); rng.shuffle(eligible)
    n=max(1,min(len(eligible)-1,round(len(eligible)*args.calibration_fraction)))
    cal=set(eligible[:n]); final=[]
    for r in clean:
        if r['group_id'] in cal:
            if not r['correct']:
                excluded.append(dict(id=r['id'],reason='failure in reserved calibration group')); continue
            r['split']='calibration'
        else: r['split']='evaluation'
        final.append(r)
    validate_records(final)
    import json
    with open(out/'cohort.jsonl','w') as f:
        for r in final: f.write(json.dumps(r)+'\n')
    write_json(out/'audit.json',dict(input_records=len(records),retained=len(final),
        calibration_groups=len(cal),excluded=excluded,
        scope='Explicit first-token or externally aligned divergence; not automatic semantic annotation'))
    print(f'Wrote {len(final)} records. Review audit.json before running models.')


if __name__=='__main__': main()

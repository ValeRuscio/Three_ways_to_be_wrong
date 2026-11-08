"""Build new, subject-disjoint PopQA recall cohorts from real model generations.
Primary protocol: full-answer alias-prefix scoring, restricted to canonical first-token
successes and unambiguous first-token failures. This is NOT semantic divergence alignment.
"""
import argparse
from collections import Counter
import csv
import json
import random
import re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from revision.common import digest, manifest, read_jsonl, seed_all, validate_records, write_json

TEMPLATES = {'capital': 'The capital of {s} is', 'capital of': '{s} is the capital of',
             'author': 'The author of {s} is', 'composer': 'The composer of {s} is',
             'director': 'The director of {s} is', 'place of birth': '{s} was born in the city of'}


def norm(s):
    s=re.sub(r'[^\w\s]', ' ', str(s).casefold())
    s=re.sub(r'\s+', ' ', s).strip()
    return re.sub(r'^(the|a|an)\s+', '', s)


def answer_matches(generation, aliases):
    text=norm(generation)
    return any(norm(a) and (text==norm(a) or text.startswith(norm(a)+' ')) for a in aliases)


def partition(group,seed):
    n=int(digest([str(group),seed])[:12],16)%100
    return 'inventory' if n<20 else 'calibration' if n<60 else 'evaluation'


def group(row): return str(row.get('subj_id',row['subj']))


def aliases_for(row):
    a=row.get('possible_answers',[])
    if isinstance(a,str): a=json.loads(a)
    return list(dict.fromkeys([str(row['obj'])]+[str(x) for x in a]))


def continuation(tok,prompt,answer):
    p=tok(prompt,add_special_tokens=True).input_ids
    whole=tok(prompt+' '+answer.strip(),add_special_tokens=True).input_ids
    if whole[:len(p)]!=p or len(whole)==len(p):
        raise ValueError('token_boundary_changes')
    return whole[len(p):]


def source_span(tok,prompt,subject,last=False):
    start=prompt.rfind(subject) if last else prompt.find(subject)
    if start<0: raise ValueError('subject_not_found')
    offsets=tok(prompt,return_offsets_mapping=True).offset_mapping
    pos=[i for i,(a,b) in enumerate(offsets) if a<start+len(subject) and b>start]
    if not pos: raise ValueError('empty_source_span')
    return [min(pos),max(pos)]


def stratified_order(rows,seed):
    rows=sorted(rows,key=lambda r:(int(r.get('s_pop') or 0),str(r['id'])))
    bins=[rows[i*len(rows)//3:(i+1)*len(rows)//3] for i in range(3)]
    rng=random.Random(seed)
    for b in bins: rng.shuffle(b)
    return [bins[j][i] for i in range(max(map(len,bins),default=0)) for j in range(3) if i<len(bins[j])]


@torch.no_grad()
def evaluate(model,tok,row,device,max_new):
    prompt=TEMPLATES[row['prop']].format(s=row['subj']); aliases=aliases_for(row)
    normalized_prompt=norm(prompt)
    if any(re.search(r'(?<!\w)'+re.escape(norm(a))+r'(?!\w)',normalized_prompt) for a in aliases if norm(a)):
        return dict(status='answer_copied_in_prompt')
    try:
        targets=continuation(tok,prompt,str(row['obj'])); g=targets[0]
        alias_first=[]
        for a in aliases:
            try: alias_first.append(continuation(tok,prompt,a)[0])
            except ValueError: pass
        span=source_span(tok,prompt,row['subj'])
    except ValueError as e: return dict(status=str(e))
    ids=tok(prompt,return_tensors='pt').input_ids.to(device)
    generated=model.generate(ids,max_new_tokens=max_new,do_sample=False,use_cache=True,
        pad_token_id=tok.eos_token_id,return_dict_in_generate=True,output_scores=True)
    new=generated.sequences[0,ids.shape[1]:]
    if not len(new): return dict(status='empty_generation')
    text=tok.decode(new,skip_special_tokens=True); top=int(new[0])
    accepted=answer_matches(text,aliases)
    audit=dict(generated=text,canonical_answer=row['obj'],aliases=aliases,top_token=top,
               canonical_first_token=g,word_level_correct=accepted)
    if accepted and top!=g: return dict(status='correct_alternative_initial_token',**audit)
    if not accepted and top==g: return dict(status='later_token_failure',**audit)
    if not accepted and top in set(alias_first): return dict(status='ambiguous_alias_initial_token',**audit)
    c=int(generated.scores[0][0].topk(2).indices[1]) if accepted else top
    record=dict(id=str(row['id']),group_id=group(row),split='unassigned',
        correct=bool(accepted),relation=row['prop'],task='recall',decision_scope='first_token',
        prompt=prompt,input_ids=ids[0].tolist(),source_span=span,target_first_token=g,
        competitor_token=c,subject=row['subj'],answer=row['obj'],aliases=aliases,
        generated=text,s_pop=int(row.get('s_pop') or 0),excluded_decoy_tokens=sorted(set(alias_first)))
    return dict(status='eligible',record=record,**audit)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True); p.add_argument('--dataset-revision',required=True)
    p.add_argument('--out',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--evaluation-per-relation',type=int,default=100)
    p.add_argument('--calibration-successes-per-relation',type=int,default=20)
    p.add_argument('--calibration-attempts-per-relation',type=int,default=250)
    p.add_argument('--decoy-pool-size',type=int,default=128); p.add_argument('--max-new-tokens',type=int,default=24)
    p.add_argument('--ranking-examples',type=int,default=60)
    args=p.parse_args(); out=manifest(args.out,args); seed_all(args.seed)
    (out/'screened').mkdir(exist_ok=True); (out/'donors').mkdir(exist_ok=True)
    from datasets import load_dataset
    dataset=load_dataset('akariasai/PopQA',split='test',revision=args.dataset_revision)
    raw=[dict(r) for r in dataset if r['prop'] in TEMPLATES]
    # One entry per subject-relation; choose a fixed dataset ID, not an outcome.
    unique={}
    for r in sorted(raw,key=lambda x:str(x['id'])): unique.setdefault((group(r),r['prop']),r)
    raw=list(unique.values())
    tok=AutoTokenizer.from_pretrained(args.model,use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,
        attn_implementation='eager').to(args.device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    inventories={}; inventory_answers={}; inventory_counts={}
    for rel in TEMPLATES:
        inventory=[r for r in raw if r['prop']==rel and partition(group(r),args.seed)=='inventory']
        answers=sorted({a for r in inventory for a in aliases_for(r)})
        token_ids=set()
        for a in answers:
            ids=tok(' '+a.strip(),add_special_tokens=False).input_ids
            if ids: token_ids.add(ids[0])
        rng=random.Random(args.seed); ordered=sorted(token_ids); rng.shuffle(ordered)
        inventories[rel]=sorted(ordered[:args.decoy_pool_size]); inventory_answers[rel]=answers
        inventory_counts[rel]=dict(subjects=len({group(r) for r in inventory}),tokens=len(inventories[rel]))
    write_json(out/'decoys.json',inventories)
    write_json(out/'inventory_audit.json',dict(counts=inventory_counts,answers=inventory_answers,
        protocol='20% subject-hash inventory, disjoint from calibration/evaluation; model-tokenized type-matched first tokens'))
    retained=[]; coverage=[]; audit_rows=[]
    for rel in TEMPLATES:
        for split in ['calibration','evaluation']:
            candidates=stratified_order([r for r in raw if r['prop']==rel and partition(group(r),args.seed)==split],args.seed)
            limit=args.calibration_attempts_per_relation if split=='calibration' else args.evaluation_per_relation
            attempts=0; correct=0; failed=0; excluded=Counter()
            for row in candidates[:limit]:
                if split=='calibration' and correct>=args.calibration_successes_per_relation: break
                attempts+=1; path=out/'screened'/f'{row["id"]}.json'
                if path.exists(): result=json.loads(path.read_text())
                else:
                    result=evaluate(model,tok,row,args.device,args.max_new_tokens); write_json(path,result)
                status=result['status']
                if status=='eligible':
                    r=result['record']; r['split']=split
                    if len(set(inventories[rel])-set(r['excluded_decoy_tokens']))<5:
                        status='fewer_than_five_non_alias_decoys'
                    elif split=='calibration' and not r['correct']: status='calibration_unsuccessful'
                    else:
                        retained.append(r); correct+=int(r['correct']); failed+=int(not r['correct'])
                if status!='eligible': excluded[status]+=1
                audit_rows.append(dict(id=str(row['id']),subject=row['subj'],relation=rel,split=split,
                    status=status,generated=result.get('generated',''),canonical_answer=row['obj']))
                if attempts%10==0: print(f'{rel}/{split}: {attempts} screened, {correct} successful, {failed} failed retained',flush=True)
            coverage.append(dict(relation=rel,split=split,available=len(candidates),screened=attempts,
                correct_retained=correct,failures_retained=failed,exclusions=dict(excluded)))
    write_json(out/'coverage.json',coverage)
    with open(out/'annotation_audit.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','subject','relation','split','status','generated','canonical_answer']); w.writeheader(); w.writerows(audit_rows)
    validate_records(retained)
    cal=[r for r in retained if r['split']=='calibration']; fails=[r for r in retained if r['split']=='evaluation' and not r['correct']]
    if len({r['group_id'] for r in cal})<10 or len({r['group_id'] for r in fails})<10:
        raise RuntimeError('Fewer than 10 independent calibration-success or evaluation-failure groups. Inspect coverage.json and use a larger cohort/model; no synthetic fallback is used.')
    def save_records(name,records):
        path=out/name; tmp=path.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r)+'\n' for r in records)); tmp.replace(path)
    save_records('cohort.jsonl',retained)
    # Balanced cycling through relations; selection independent of attribution scores.
    byrel={rel:[r for r in fails if r['relation']==rel] for rel in TEMPLATES}
    rng=random.Random(args.seed)
    for rs in byrel.values(): rng.shuffle(rs)
    chosen=[rs[i] for i in range(max(map(len,byrel.values()),default=0)) for rs in byrel.values() if i<len(rs)][:args.ranking_examples]
    save_records('ranking_cohort.jsonl',cal+chosen)
    donor_records=[]; donor_audit=[]
    for r in chosen:
        path=out/'donors'/f'{r["id"]}.json'
        if path.exists(): result=json.loads(path.read_text())
        else:
            donor=f'Information: {r["subject"]} — {r["relation"]}: {r["answer"]}.\n{r["prompt"]}'
            ids=tok(donor,return_tensors='pt').input_ids.to(args.device)
            with torch.no_grad(): donor_top=int(model(ids,use_cache=False).logits[0,-1].argmax())
            span=source_span(tok,donor,r['subject'],last=True)
            okay=donor_top==r['target_first_token'] and span[1]-span[0]==r['source_span'][1]-r['source_span'][0]
            result=dict(eligible=okay,top_token=donor_top,span=span,input_ids=ids[0].tolist())
            if okay:
                alternatives=[a for a in inventory_answers[r['relation']] if norm(a) not in {norm(x) for x in r['aliases']}]
                for wrong_answer in alternatives[:20]:
                    wrong=f'Information: {r["subject"]} — {r["relation"]}: {wrong_answer}.\n{r["prompt"]}'
                    wi=tok(wrong,return_tensors='pt').input_ids.to(args.device)
                    with torch.no_grad(): wt=int(model(wi,use_cache=False).logits[0,-1].argmax())
                    ws=source_span(tok,wrong,r['subject'],last=True)
                    if wt!=r['target_first_token'] and ws[1]-ws[0]==span[1]-span[0]:
                        result.update(wrong_input_ids=wi[0].tolist(),wrong_span=ws); break
            write_json(path,result)
        donor_audit.append(dict(id=r['id'],eligible=result['eligible'],wrong_donor_available='wrong_input_ids' in result))
        if result['eligible']:
            d=dict(r,donor_input_ids=result['input_ids'],donor_source_span=result['span'])
            if 'wrong_input_ids' in result: d.update(wrong_donor_input_ids=result['wrong_input_ids'],wrong_donor_source_span=result['wrong_span'])
            donor_records.append(d)
    save_records('internal_cohort.jsonl',cal+donor_records)
    write_json(out/'donor_coverage.json',donor_audit)
    write_json(out/'protocol.json',dict(dataset='akariasai/PopQA',dataset_revision=args.dataset_revision,
        model=args.model,selection='Subject hash: 20% inventory, 40% calibration, 40% evaluation. Candidate order interleaves popularity tertiles.',
        endpoint='Canonical first token; automatic alias-prefix full-answer scoring; alternative-alias initials and later-token failures excluded.',
        limitation='New restricted cohort, not reproduction of the paper; manually inspect annotation_audit.csv before publication.',
        calibration_successes=len(cal),evaluation_failures=len(fails),ranking_examples=len(chosen),verified_donors=len(donor_records)))
    print(json.dumps(dict(calibration=len(cal),evaluation_failures=len(fails),verified_donors=len(donor_records)),indent=2))


if __name__=='__main__': main()

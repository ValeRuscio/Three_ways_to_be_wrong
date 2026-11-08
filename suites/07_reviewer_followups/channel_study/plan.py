import argparse,json
from .io import load
from .run import subset

def plan(cfg,prompts,stages):
 rows=[json.loads(s) for s in open(prompts) if s.strip()]
 rows=subset([r for r in rows if r['template'] in cfg['templates']],cfg['max_prompts'],cfg['seed'])
 n=len(rows);facts=len(set(r['id'] for r in rows));probes=min(facts,cfg['probe_facts']) if cfg['probe_facts'] is not None else facts
 layers=len(cfg['probe_layers']) if cfg['probe_layers'] is not None else len(set(cfg['probe_layer_fractions']))
 perturb=probes*layers*len(cfg['families'])*len(cfg['seeds'])*len(cfg['doses'])
 frontier=probes*layers*len(cfg['frontier_families'])*(3+len(cfg['seeds']))*len(cfg['doses'])
 print('Models:',[s['key'] for s in cfg['models']]);print('Requested stages:',stages)
 print(f'Full selected cohort: {n} prompts / {facts} facts. No correctness-based selection.')
 if 'split' in stages:print(f'Split: {2*n:,} backward passes per model (plus capture/generation).')
 if 'fisher' in stages:print(f'Fisher: {2*n*cfg["fisher_draws"]:,} EXTRA backward passes per model.')
 if 'geometry' in stages:print(f'Geometry: {n:,} origin decompositions per model plus CPU subspace fits.')
 if 'perturb' in stages:print(f'Perturb: {perturb:,} random edits per model, each with live and frozen forwards.')
 if 'frontier' in stages:
  print(f'Frontier: at most {frontier:,} edits per model (zero gradient directions skipped).')
  print(f'Each edit generates target + up to {cfg["specificity_n"] if cfg["control_generation"] else 0} control answers, up to {cfg["max_new_tokens"]} new tokens each.')
 if 'training' in stages:print('Training:',[s['key'] for s in cfg['training_models']],'; up to',cfg['training_facts'],'prespecified facts per checkpoint; fractions',cfg['training_fractions'])
 print('FP32 eager attention. No validated GPU runtime estimate. Stage timings are printed during execution.')
 print('Hypothesis tests are NOT implied by a positive descriptive mean. No answer-site-only interpretation of global weight gradients.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--prompts',required=True);p.add_argument('--stages',nargs='+',required=True);a=p.parse_args();plan(load(a.config),a.prompts,a.stages)

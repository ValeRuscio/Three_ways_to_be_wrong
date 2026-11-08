import argparse
from pathlib import Path
from channel_study.io import load
from .data import prepare

def plan(cfg,out,stages,previous=None):
 cohort=prepare(cfg,out,previous);ev=cohort['evaluation'];cal=cohort['calibration']
 print('Evaluation:',len(ev),'prompts,',len({r['id'] for r in ev}),'facts. Calibration:',len(cal),'prompts.')
 print('Models:',[s['key'] for s in cfg['models']]);print('Stages:',stages)
 print('Previous results:',previous or 'none; fresh data')
 if 'profiles' in stages:print('Profiles: 2 backward passes per uncached prompt/model; compatible previous splits reused.')
 if set(stages)&{'regimes','variance','sink'}:print('Capture/origins: new forward/origin propagation on evaluation prompts; regimes also screen independent calibration prompts. No cached gates are assumed.')
 if 'nulls' in stages:print('Nulls:',cfg['null_prompts'],'matched prompts on',cfg['null_models'],'; random initialization seed',cfg['random_initialization_seed'],'; random readout seeds',cfg['readout_seeds'],'. Two backwards per condition/draw/prompt.')
 if 'freeze' in stages:print('Three mechanisms:',cfg['freeze_prompts'],'prompts on',cfg['freeze_models'],'; 8 gradient passes per prompt for exact Shapley attribution (not 3).')
 if 'matrix' in stages:print('Layer matrix:',cfg['matrix_prompts'],'prompts on',cfg['matrix_models'],'; L+2 backward passes per prompt, all source layers in each pass.')
 if 'certificate' in stages:print('Certificates:',cfg['certificate_prompts'],'prompts/model; 3 layer fractions, 7 families,',len(cfg['seeds'])+1,'random/aligned directions,',len(cfg['fd_epsilons']),'epsilon values; each check uses 3 forward evaluations.')
 print('Null/freeze/matrix/certificate subsets are chosen by hash BEFORE observing outcomes. Change *_prompts to None for full cohorts before first run.')
 print('No validated GH200 runtime estimate. Source-layer matrix and Shapley stages can take hours. Run stages separately; completed work resumes.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--previous');p.add_argument('--stages',nargs='+',required=True);a=p.parse_args();plan(load(a.config),a.out,a.stages,a.previous)

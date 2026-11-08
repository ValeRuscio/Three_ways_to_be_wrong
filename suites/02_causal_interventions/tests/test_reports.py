import json,copy
from pathlib import Path
from causal_suite.data import save
from causal_suite.config import preset,spec
from causal_suite.measure import assign_labels
from causal_suite.report import report


def record(i,split='evaluation',correct=False):
 origins={'subject':float(i%5+1),'template':2.,'first_token':1.,'bias':0.}
 led=dict(origins=origins,logit=sum(origins.values()),handoff=[0,.2,origins['subject']],total_by_cut=[origins['subject']]*3,reconstruction_error=0,handoff_error=0,random_span_support=.1)
 return dict(id=str(i),group_id=str(i),subject='S'+str(i),relation='capital',answer='A',aliases=['A'],split=split,correct=correct,
  eligible=True,first_token_correct=correct,canonical_first_correct=correct,margin=1. if correct else -1.,tau=origins['subject'],share=.5,competitor_share=.4+i*.001,competitor_absolute_share=.4,
  q={'full':-1. if i%3==0 else 1.,'hard20':1.},q_max={'full':1.,'hard20':2.},g=1,c=2,replaced_competitor=3,random_span=[0],
  ledger={'1':copy.deepcopy(led),'2':copy.deepcopy(led),'3':copy.deepcopy(led)},certificate=0.)

def test_reports_full_workflow(tmp_path):
 cfg=preset();cfg.update(bootstrap=12,permutations=12,min_calibration=2,calibration_reference_n=8)
 models=[spec('base','unused','pair','base'),spec('instruct','unused','pair','instruct')]+[spec('step'+str(i),'unused','train','checkpoint',step=i) for i in [100,200,300]]
 for m in models:
  records=[record(i,'calibration',True) for i in range(100,108)]+[record(i,correct=(i%4==0 or (m['stage']=='instruct' and i%4==1) or (m['step']==300 and i%4==2))) for i in range(12)]
  records=assign_labels(records,cfg);d=tmp_path/m['key']/'raw';save(d/'model.json',m);save(d/'measurements.json',records)
  repair=[]
  for r in records[8:]:
   for family,arms in [('source',['donor','random','wrong_subject']),('transport',['top_heads','random_span','bottom_heads']),('selection',['alternative_down','answer_up','random'])]:
    for arm in arms:
     for dose in [.5,1.,2.]:repair.append(dict(id=r['id'],group_id=r['group_id'],relation='capital',label=r['labels']['full'],correct_before=r['correct'],family=family,arm=arm,dose=dose,layer=1 if family=='source' else None,margin_gain=dose*.1,recovered=dose==2.,final_residual_relative_cost=dose*.1,final_normalized_relative_cost=dose*.1,plant='natural'))
  save(d/'repair_records'/'x.json',dict(id='x',donor_available=True,wrong_donor_available=True,rows=repair))
 report(tmp_path,cfg)
 for name in ['origin_auc_holm.csv','alignment_transitions.csv','training_trends.csv','repair_control_comparisons_holm.csv','repair_pooled_fixed_effects_holm.csv','interpretation.json']:
  assert (tmp_path/'reports'/name).exists()

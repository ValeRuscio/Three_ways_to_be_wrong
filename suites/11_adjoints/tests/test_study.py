import tempfile,unittest,json
from pathlib import Path
import numpy as np
import torch
from transformers import LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM,Olmo2Config,Olmo2ForCausalLM
from causal_suite.engine import Engine
from followup_study.mechanisms import capture,MECHANISMS
from adjoint_study.core import *
from adjoint_study.report import report

torch.set_num_threads(1)
CFG=json.loads((Path(__file__).parents[1]/'adjoint_study/config.json').read_text())

def tiny(kind='llama'):
    torch.manual_seed(12)
    cls,mdl={'llama':(LlamaConfig,LlamaForCausalLM),'qwen2':(Qwen2Config,Qwen2ForCausalLM),'olmo2':(Olmo2Config,Olmo2ForCausalLM)}[kind]
    cfg=cls(vocab_size=31,hidden_size=12,intermediate_size=16,num_hidden_layers=2,num_attention_heads=3,num_key_value_heads=1,max_position_embeddings=64)
    cfg._attn_implementation='eager';e=Engine(mdl(cfg).float());C=capture(e,torch.tensor([1,4,6,8,2]),1e-4);return e,C

class Tests(unittest.TestCase):
    def test_adjoint_engine_and_fd(self):
        for kind in ['llama','qwen2','olmo2']:
            e,C=tiny(kind);r,L,F=measure_adjoints(e,C,7,3,[1,2])
            x=C['layers'][0]['x'].detach().requires_grad_();y=x
            for l in range(e.L):y=e.layer(y,l,C)
            z=e.read(y,C);d=torch.autograd.grad(z[7]-z[3],x)[0]
            torch.testing.assert_close(F[0],d,rtol=2e-4,atol=2e-5)
            checks=embedding_checks(e,C,7,3,[1,2],L,F,dict(CFG,fd_epsilons=[.001],fd_atol=2e-6,fd_rtol=.03))
            self.assertTrue(all(v['passed'] for v in checks));self.assertTrue(any(v['resolved'] for v in checks))
            self.assertEqual(L.shape,(3,5,12));self.assertLess(r['baseline_error'],1e-5)
            self.assertLess(max(v['closure_error'] for v in r['roles']),1e-10)
    def test_jacobian_estimator(self):
        e,C=tiny();cfg=dict(CFG,jacobian_probes=1024,crossing_run=2)
        result=gate_jacobian(e,C,[1,2],cfg)
        x=e.model.get_input_embeddings()(C['ids'][None]).detach().requires_grad_();saved=[]
        h=e.blocks[0].mlp.act_fn.register_forward_hook(lambda m,a,o:saved.append(o[0,-1]))
        try:
            with embedding_input(e,x):live(e,C['ids'])
        finally:h.remove()
        J=torch.stack([torch.autograd.grad(saved[0][i],x,retain_graph=True)[0][0] for i in range(16)])
        for role,idx in groups(5,[1,2]).items():
            if role not in ['subject','template','first']:continue
            exact=float(J[:,idx].double().square().sum());estimate=next(r['jacobian_sq'] for r in result['rows'] if r['layer']==0 and r['role']==role)
            self.assertAlmostEqual(estimate/exact,1.,delta=.15)
    def test_sgd_and_restoration(self):
        for kind in ['llama','qwen2','olmo2']:
            e,C=tiny(kind);before={k:p.clone() for k,p in e.model.named_parameters()}
            result=sgd_experiment(e,C,7,3,dict(CFG,weight_layers=[0,1],weight_families=['gate','up','down','o','v','q','k'],step_doses=[1e-4]))
            self.assertTrue(all(v['frozen_linear_pass'] for v in result))
            self.assertTrue(all(v['ce_after']<=v['baseline_ce']+1e-6 for v in result))
            for k,p in e.model.named_parameters():self.assertTrue(torch.equal(p,before[k]))
            for row in result:
                self.assertAlmostEqual(row['live_gain'],row['frozen_gain']+row['finite_reselection'])
            gate=[r for r in result if r['family']=='gate'];self.assertTrue(all(abs(r['frozen_gain'])<1e-6 for r in gate))
    def test_hessian_and_restoration(self):
        e,C=tiny();result=curvature(e,C,7,3,dict(CFG,weight_layers=[0],weight_families=['down','gate'],curvature_epsilons=[.01]))
        self.assertTrue(all(abs(v['frozen_directional_hessian'])<1e-7 for v in result))
        self.assertTrue(all(v['passed'] for v in result))
    def test_groups_and_report(self):
        self.assertEqual(groups(3,[0,1])['first'],[])
        self.assertEqual(set(sum([groups(5,[1,2])[k] for k in ['subject','first','template']],[])),set(range(5)))
        e,C=tiny();data,_,_=measure_adjoints(e,C,7,3,[1,2]);data['finite_checks']=[]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'llama1b/adjoints/x.json';p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(model='llama1b',prompt_key='x',id='a',group_id='a',relation='capital',template=0,regime='uncalibrated_failure',seconds=1,baseline_margin=0,native_certificate=C['certificate'],data=data)))
            report(root,dict(CFG,models=['llama1b'],bootstrap=10))
            self.assertTrue((root/'reports/adjoint_summary.csv').exists());self.assertTrue((root/'reports/input_origin_summary.csv').exists())
    def test_runner_resume_all_stages(self):
        from adjoint_study.run import main_run
        from unittest.mock import patch
        e,C=tiny()
        row=dict(id='1',group_id='s1',template=0,relation='capital',prompt='Some subject is',subject='subject',answer='answer',aliases=['answer'])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);evidence=root/'evidence';(evidence/'data').mkdir(parents=True);(evidence/'llama1b').mkdir()
            (evidence/'data/cohort.json').write_text(json.dumps(dict(evaluation=[row],calibration=[])))
            (evidence/'llama1b/model.json').write_text(json.dumps(dict(repo='test',sha='test')))
            (evidence/'regimes.csv').write_text('model,id,template,regime\nllama1b,1,0,uncalibrated_failure\n')
            out=root/'result';out.mkdir()
            cfg=dict(CFG,models=['llama1b'],stages=['adjoints','jacobian','sgd','curvature'],min_free_gb=0,
                jacobian_probes=2,bootstrap=10,fd_prompts=0,step_doses=[.0001],curvature_epsilons=[.001])
            with patch('adjoint_study.run.model_load',return_value=(None,e)) as loader,patch('adjoint_study.run.tokenize',return_value=(C['ids'],[1,2],7)):
                main_run(cfg,out,evidence);calls=loader.call_count
                main_run(cfg,out,evidence);self.assertEqual(loader.call_count,calls)
                self.assertTrue((out/'completion.json').exists())
                self.assertTrue((out/'reports/sgd_summary.csv').exists())
                self.assertTrue((out/'reports/crossover_within_relation.csv').exists())
                with self.assertRaises(RuntimeError):main_run(dict(cfg,seed=999),out,evidence)
if __name__=='__main__':unittest.main()

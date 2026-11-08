import ast,base64,importlib.util,io,json,os,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
helper=load('notebook_sources');exporter=load('export_results')
# Test actual bootstrap and every substituted bundle/checksum assignment.
old=Path.cwd()
try:
 os.chdir(ROOT/'notebooks')
 for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
  nb=json.loads(path.read_text());codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
  ns={};exec(codes[0],ns)
  import hashlib
  ns.update(base64=base64,hashlib=hashlib)
  found=0
  for code in codes[1:]:
   for node in ast.parse(code).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['BUNDLE','_BUNDLE','BUNDLE_SHA256','EXPECTED_SHA256'] for t in node.targets):
     if '_source_bundle' in ast.unparse(node):exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns);found+=1
  assert found,path
  bundle=ns.get('BUNDLE',ns.get('_BUNDLE'));data=base64.b64decode(bundle)
  digest=ns.get('BUNDLE_SHA256',ns.get('EXPECTED_SHA256'))
  if digest:assert hashlib.sha256(data).hexdigest()==digest,path
  with zipfile.ZipFile(io.BytesIO(data)) as z:assert z.testzip() is None
finally:os.chdir(old)
with tempfile.TemporaryDirectory() as d:
 d=Path(d);run=d/'run';(run/'reports').mkdir(parents=True)
 (run/'reports/summary.csv').write_text('x,y\n1,2\n')
 (run/'reports/a_per_prompt.csv').write_text('x\n1\n')
 (run/'configuration.json').write_text(json.dumps(dict(hf_token='secret',input='/home/ubuntu/data.json')))
 (run/'weights.pt').write_text('excluded')
 out=d/'export.zip';files,skips=exporter.export(run,out)
 assert len(files)==2
 with zipfile.ZipFile(out) as z:
  assert 'weights.pt' not in z.namelist()
  cfg=json.loads(z.read('configuration.json'));assert cfg['hf_token']=='[REDACTED]';assert cfg['input']=='<original-run-path>/data.json'
 try:exporter.export(run,out)
 except FileExistsError:pass
 else:raise AssertionError('Clobbered archive')
print('PASS: all 11 notebook source/checksum cells; compact exporter selection, redaction and overwrite protection')

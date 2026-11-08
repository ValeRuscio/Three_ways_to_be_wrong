"""Offline syntax, notebook, file-integrity and bundle checks. No GPU/downloads."""
import ast,hashlib,json,sys,base64,io,zipfile
from pathlib import Path
from notebook_sources import source_bundle
ROOT=Path(__file__).resolve().parents[1]
def main():
    count=0
    for p in ROOT.rglob('*.py'):
        if '__pycache__' not in p.parts:ast.parse(p.read_text(),filename=str(p));count+=1
    for p in (ROOT/'notebooks').glob('*.ipynb'):
        nb=json.loads(p.read_text())
        for c in nb['cells']:
            if c['cell_type']=='code':
                assert not c.get('outputs'),str(p)+' has saved outputs'
                ast.parse(''.join(c['source']),filename=str(p))
        # The first code cell must locate the repository and import its source helper.
        source=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        assert 'notebook_sources' in source
    for suite in (ROOT/'suites').iterdir():
        if suite.is_dir():
            b=source_bundle(suite.name)
            assert b==source_bundle(suite.name),'Non-deterministic source bundle'
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(b))) as z:assert z.testzip() is None
    manifest=ROOT/'MANIFEST.json'
    if manifest.exists():
        for name,digest in json.loads(manifest.read_text()).items():
            p=ROOT/name
            assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,'Changed/missing release file: '+name
    print(f'PASS: {count} Python files, notebook syntax, deterministic bundles and release manifest')
if __name__=='__main__':main()

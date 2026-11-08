"""Export a compact review/publication subset from an existing run. Read-only input.

Example:
  python supplementary/tools/export_results.py /path/to/run --out /path/to/new_export.zip

Includes report CSV/JSON/PDF/PNG, configurations, checkpoint metadata and completion
records. Excludes per-prompt/per-check rows unless explicitly requested. Does not
export logs, source caches, tensors or model weights. A snapshot is not resumable.
"""
import argparse,hashlib,json,zipfile
from pathlib import Path

METADATA={'configuration.json','environment.json','model.json','architecture.json',
          'completion.json','exit.json'}

def cleaned(data):
    obj=json.loads(data)
    def visit(x):
        if isinstance(x,dict):return {k:('[REDACTED]' if k.lower() in {'token','hf_token','access_token','password','api_key'} else visit(v)) for k,v in x.items()}
        if isinstance(x,list):return [visit(v) for v in x]
        if isinstance(x,str) and (x.startswith('/home/') or x.startswith('/Users/')):return '<original-run-path>/'+Path(x).name
        return x
    return (json.dumps(visit(obj),indent=2)+'\n').encode()

def export(root,out,include_rows=False,max_mib=25):
    root=Path(root).expanduser().resolve();out=Path(out).expanduser().resolve()
    if not root.is_dir():raise ValueError('Run directory does not exist: '+str(root))
    if out.exists():raise FileExistsError('Choose a new export filename; existing exports are preserved')
    paths=[];skipped=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.is_symlink() or p==out:continue
        rel=p.relative_to(root)
        if any(x in rel.parts for x in ['.git','__pycache__','snapshots']):continue
        report='reports' in rel.parts and p.suffix in {'.csv','.json','.pdf','.png'}
        if not (report or p.name in METADATA):continue
        if not include_rows and any(x in p.name for x in ['per_prompt','per_check','per_fact','per_subject']):continue
        if p.stat().st_size>max_mib*2**20:skipped.append(dict(path=str(rel),reason='size limit'));continue
        paths.append(p)
    if not paths:raise ValueError('No supported reports or metadata found; check the run path')
    out.parent.mkdir(parents=True,exist_ok=True);manifest=[]
    # Exclusive creation prevents clobbering an existing export.
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            try:
                data=p.read_bytes()
                if p.suffix=='.json':data=cleaned(data)
            except (OSError,ValueError) as e:
                skipped.append(dict(path=str(p.relative_to(root)),reason=type(e).__name__));continue
            name=p.relative_to(root).as_posix();z.writestr(name,data)
            manifest.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        z.writestr('EXPORT_MANIFEST.json',json.dumps(dict(
            kind='compact_results_subset',resumable=False,
            completion='Inspect included exit/completion records; export alone does not establish completion.',
            consistency='For a completed run, export after all workers exit. Active runs are non-atomic snapshots.',
            files=manifest,skipped=skipped),indent=2))
    return manifest,skipped

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--include-rows',action='store_true');p.add_argument('--max-file-mib',type=float,default=25)
    a=p.parse_args();files,skipped=export(a.run,a.out,a.include_rows,a.max_file_mib)
    print(f'Exported {len(files)} files; skipped {len(skipped)}. {a.out} ({a.out.stat().st_size/2**20:.1f} MiB)')
if __name__=='__main__':main()

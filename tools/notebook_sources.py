"""Build deterministic notebook launch bundles from reviewable repository files."""
import base64,io,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def source_bundle(suite):
    path=ROOT/'suites'/suite
    if path.parent != ROOT/'suites' or not path.is_dir():
        raise ValueError('Unknown suite: '+suite)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(path.rglob('*')):
            if not p.is_file() or any(x in p.parts for x in ['__pycache__','.pytest_cache','.git']):continue
            if p.suffix in ['.pyc','.pyo']:continue
            info=zipfile.ZipInfo(p.relative_to(path).as_posix(),date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,p.read_bytes())
    return base64.b64encode(buffer.getvalue()).decode()

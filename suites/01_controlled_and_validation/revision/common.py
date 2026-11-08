import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import numpy as np


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def manifest(out, args, inputs=()):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    files = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs}
    root = Path(__file__).resolve().parent.parent
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(root.glob('*.py')) + sorted((root/'revision').rglob('*.py'))}
    signature = digest({'config': config, 'inputs': files, 'source': source_hashes})
    path = out / 'manifest.json'
    if path.exists() and json.loads(path.read_text())['signature'] != signature:
        raise ValueError('Output directory belongs to another configuration. Use a new --out.')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    versions = {}
    for name in ['torch', 'transformers', 'numpy', 'scipy', 'pandas', 'scikit-learn']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    path.write_text(json.dumps(dict(signature=signature, config=config, inputs=files,
                                   git_commit=commit, source_hashes=source_hashes, python=platform.python_version(),
                                   packages=versions), indent=2))
    return out


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(tmp, path)


def seed_all(seed):
    import random
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def label(readable, tau, share, tau_threshold, share_threshold):
    if not readable: return 'source'
    if tau < tau_threshold or share < share_threshold: return 'transport'
    return 'selection'


def run_score(q, consecutive=2):
    """q [position, layer]. Max over windows of their minimum; >0 iff readable."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] < consecutive or consecutive < 1:
        raise ValueError('Not enough layers for readability window')
    return float(max(q[:, i:i+consecutive].min(axis=1).max()
                     for i in range(q.shape[1]-consecutive+1)))


def validate_records(records):
    ids = [str(r['id']) for r in records]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate id')
    groups = {}
    for r in records:
        if r['split'] not in ('calibration', 'evaluation'): raise ValueError('Invalid split')
        if not isinstance(r['correct'], bool): raise ValueError('correct must be a JSON boolean')
        a, b = r['source_span']
        if not (isinstance(a, int) and isinstance(b, int) and 0 <= a <= b):
            raise ValueError('source_span must contain inclusive token indices')
        group = str(r['group_id'])
        groups.setdefault(group, set()).add(r['split'])
        if r['split'] == 'calibration' and not r['correct']:
            raise ValueError('Calibration records must be successful examples')
    if any(len(v) > 1 for v in groups.values()):
        raise ValueError('A group occurs in calibration and evaluation: leakage')
    if not any(r['split']=='calibration' for r in records): raise ValueError('No calibration records')
    if not any(r['split']=='evaluation' for r in records): raise ValueError('No evaluation records')


def bootstrap_mean(values, groups, n_boot=1000, seed=0):
    """Cluster-resampled CI for the example-weighted mean."""
    values = np.asarray(values, dtype=float); groups = np.asarray(groups).astype(str)
    keep = np.isfinite(values); values, groups = values[keep], groups[keep]
    unique = np.unique(groups)
    if not len(unique): return dict(mean=None, lo=None, hi=None, n=0)
    rng = np.random.default_rng(seed)
    blocks = [values[groups == g] for g in unique]
    means = [np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))]).mean()
             for _ in range(n_boot)]
    return dict(mean=float(values.mean()), lo=float(np.quantile(means,.025)),
                hi=float(np.quantile(means,.975)), n=len(values), n_groups=len(unique))

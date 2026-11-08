"""Explicit, locked migration from the exact delivered original implementation."""
from pathlib import Path
import shutil
from .common import load,dump

ORIGINAL_HASH='2af239ee17abee1f33aecf54'


def migrate(out,expected,source_root):
 out=Path(out);path=out/'protocol.json';old=load(path)
 if old==expected:return False
 if old.get('code_hash')!=ORIGINAL_HASH or {k:v for k,v in old.items() if k!='code_hash'}!={k:v for k,v in expected.items() if k!='code_hash'}:
  raise ValueError('Fast migration only accepts the original delivered code with IDENTICAL configuration and input. No results have been changed.')
 dump(out/'protocol_before_fast.json',old)
 archive=out/'source_versions'/ORIGINAL_HASH;archive.parent.mkdir(parents=True,exist_ok=True)
 if (out/'source').exists() and not archive.exists():shutil.move(str(out/'source'),str(archive))
 # Idempotent if an earlier migration was interrupted during source copying.
 for name in ['pattern_study','gate_study','causal_suite']:
  shutil.copytree(Path(source_root)/name,out/'source'/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 dump(out/'fast_upgrade.json',dict(previous_code_hash=old['code_hash'],new_code_hash=expected['code_hash'],scope='Final-token curve evaluation only; all samples, seeds, doses, endpoints and float32 precision retained',completed_files='Preserved; original source and protocol archived',validation='Per-recipient native and perturbed equivalence checks, with original-path fallback'))
 dump(path,expected)
 return True

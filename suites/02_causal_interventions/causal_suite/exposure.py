"""Optional streaming document co-occurrence counts from YOUR corpus shard(s).
Counts are document-level lexical matches, not causal or checkpoint-specific
training exposure. Do not call a subset of the Pile the full Pile.
python -m causal_suite.exposure --facts results/causal_v1/fixed_data.json \
  --corpus corpus.jsonl --out exposure.csv
Each corpus line is a JSON object with a 'text' field; .gz accepted.
"""
import argparse,json,gzip,re,hashlib
from pathlib import Path
import pandas as pd
from .data import save


def count(facts,corpora):
 rows=[r for r in facts if r['split']=='evaluation'];terms={r['subject'].casefold() for r in rows}|{a.casefold() for r in rows for a in r['aliases']}
 terms.discard('');patterns={term:re.compile(r'(?<!\w)'+re.escape(term)+r'(?!\w)',re.I) for term in terms}
 counts={r['id']:0 for r in rows};documents=0;digests={}
 for path in corpora:
  path=Path(path);hasher=hashlib.sha256()
  with open(path,'rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):hasher.update(chunk)
  digests[str(path)]=hasher.hexdigest()
  opener=gzip.open if path.suffix=='.gz' else open
  with opener(path,'rt') as f:
   for line in f:
    if not line.strip():continue
    document=json.loads(line);text=document['text'];found={term for term,pattern in patterns.items() if pattern.search(text)};documents+=1
    for r in rows:
     if r['subject'].casefold() in found and any(a.casefold() in found for a in r['aliases']):counts[r['id']]+=1
 return [dict(id=i,exposure=n) for i,n in counts.items()],dict(documents=documents,corpus_sha256=digests,definition='Document-level subject + any answer alias co-occurrence; lexical matching; supplied corpus only')


def main():
 p=argparse.ArgumentParser();p.add_argument('--facts',required=True);p.add_argument('--corpus',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 facts=json.loads(Path(a.facts).read_text())['rows'];rows,meta=count(facts,a.corpus);pd.DataFrame(rows).to_csv(a.out,index=False);save(str(a.out)+'.metadata.json',meta)
if __name__=='__main__':main()

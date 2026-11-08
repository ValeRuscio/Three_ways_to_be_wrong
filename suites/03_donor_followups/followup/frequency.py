"""Count a user-supplied corpus. Corpus counts are a proxy unless it is the
actual training distribution. Never infer frequencies from PopQA popularity.
"""
import argparse,json,hashlib,gzip
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from causal_suite.data import digest
from .common import dump


def tokenizer_hash(tok):return digest(sorted(tok.get_vocab().items()))


def main():
 p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--model',required=True);p.add_argument('--revision',required=True);p.add_argument('--out',required=True);p.add_argument('--text-key',default='text');p.add_argument('--vocab-size',type=int);p.add_argument('--description',required=True);a=p.parse_args()
 tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision);counts=np.zeros(a.vocab_size or len(tok),dtype=np.int64);path=Path(a.corpus);sha=hashlib.sha256();documents=0
 opener=gzip.open if path.suffix=='.gz' else open
 with opener(path,'rt',encoding='utf8') as f:
  for line in f:
   sha.update(line.encode());obj=json.loads(line);text=obj[a.text_key]
   if not isinstance(text,str):raise ValueError('Corpus text must be a string')
   ids=tok.encode(text,add_special_tokens=False)
   if ids and max(ids)>=len(counts):raise ValueError('Set --vocab-size to model output vocabulary size')
   np.add.at(counts,ids,1);documents+=1
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 with open(out,'wb') as f:np.savez_compressed(f,counts=counts)
 dump(out.with_suffix('.json'),dict(tokenizer_hash=tokenizer_hash(tok),model=a.model,revision=a.revision,corpus_sha256=sha.hexdigest(),documents=documents,tokens=int(counts.sum()),description=a.description,interpretation='Frequency on supplied corpus; not automatically pretraining frequency'))
 print(out)

if __name__=='__main__':main()

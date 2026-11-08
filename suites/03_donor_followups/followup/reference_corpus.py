"""Build independent WikiText-103 unigram proxies for all input models.
This is NOT reconstruction of their pretraining token frequencies.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer,AutoConfig
from huggingface_hub import HfApi
from .common import sources,dump
from .frequency import tokenizer_hash


def load_dataset(*args,**kwargs):
 from datasets import load_dataset as loader
 return loader(*args,**kwargs)


def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
 _,ss=sources(a.input,out/'inputs');pin=out/'dataset.json'
 if pin.exists():revision=json.loads(pin.read_text())['revision']
 else:
  revision=HfApi().dataset_info('Salesforce/wikitext').sha
  dump(pin,dict(dataset='Salesforce/wikitext',config='wikitext-103-raw-v1',split='train',revision=revision))
 data=None;mapping={}
 for source in ss:
  spec=source['spec'];key=spec['key'];final=out/(key+'.npz');meta_path=final.with_suffix('.json');mapping[key]=str(final)
  if final.exists() and meta_path.exists():
   meta=json.loads(meta_path.read_text())
   if meta['model_sha']!=spec['sha'] or meta['dataset_revision']!=revision:raise ValueError('Frequency checkpoint/dataset changed; use a new directory')
   if hashlib.sha256(final.read_bytes()).hexdigest()!=meta['counts_sha256']:raise ValueError('Frequency cache changed')
   continue
  tok=AutoTokenizer.from_pretrained(spec['repo'],revision=spec['sha']);vocab=AutoConfig.from_pretrained(spec['repo'],revision=spec['sha']).vocab_size
  if data is None:data=load_dataset('Salesforce/wikitext','wikitext-103-raw-v1',revision=revision,split='train')
  checkpoint=out/(key+'_partial.npz');counts=np.zeros(vocab,dtype=np.int64);start=0
  if checkpoint.exists():
   with np.load(checkpoint,allow_pickle=False) as z:
    if str(z['model_sha'])!=spec['sha'] or str(z['dataset_revision'])!=revision:raise ValueError('Partial frequency checkpoint mismatch')
    counts=z['counts'];start=int(z['next_row'])
  for i in range(start,len(data),512):
   texts=data[i:i+512]['text'];encoded=tok(texts,add_special_tokens=False,truncation=False)['input_ids'];flat=[v for row in encoded for v in row]
   if flat and max(flat)>=vocab:raise ValueError('Tokenizer IDs exceed output vocabulary')
   counts+=np.bincount(flat,minlength=vocab)
   if (i-start)//512%200==0 or i+512>=len(data):
    tmp=checkpoint.with_suffix('.tmp')
    with open(tmp,'wb') as f:np.savez_compressed(f,counts=counts,next_row=min(i+512,len(data)),model_sha=spec['sha'],dataset_revision=revision)
    tmp.replace(checkpoint);print(f'{key}: {min(i+512,len(data))}/{len(data)} WikiText rows counted',flush=True)
  temp=final.with_suffix('.tmp')
  with open(temp,'wb') as f:np.savez_compressed(f,counts=counts)
  temp.replace(final)
  dump(meta_path,dict(tokenizer_hash=tokenizer_hash(tok),model_sha=spec['sha'],dataset_revision=revision,source='Salesforce/wikitext wikitext-103-raw-v1 train',source_url='https://huggingface.co/datasets/Salesforce/wikitext',tokens=int(counts.sum()),rows=len(data),counts_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),interpretation='Independent corpus frequency proxy; not the actual model pretraining distribution. Rows tokenized separately without special tokens.'))
  checkpoint.unlink(missing_ok=True)
 dump(out/'mapping.json',mapping);print('Frequency mapping:',out/'mapping.json')

if __name__=='__main__':main()

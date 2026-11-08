"""Create a tiny random model and synthetic test cohort. NOT scientific evidence."""
import argparse
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast
from revision.common import seed_all,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True)
    args=p.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    seed_all(7); torch.set_num_threads(1)
    vocab={'<unk>':0,**{f'w{i}':i for i in range(1,32)}}
    t=Tokenizer(WordLevel(vocab,unk_token='<unk>')); t.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=t,unk_token='<unk>')
    model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        max_position_embeddings=32,attn_implementation='eager')).eval()
    model.save_pretrained(out/'model'); tok.save_pretrained(out/'model')
    rows=[]
    for i in range(60):
        ids=torch.randint(1,32,(6,))
        with torch.no_grad(): z=model(ids[None]).logits[0,-1]
        order=z.argsort(descending=True); success=i<20 or i%5==0
        g=int(order[0] if success else order[1+(i%10)])
        rows.append(dict(id=str(i),group_id=f'fact{i}',split='calibration' if i<20 else 'evaluation',
            correct=success,relation=f'relation{i%2}',task='smoke',decision_scope='first_token',
            input_ids=ids.tolist(),prompt=' '.join(f'w{j}' for j in ids.tolist()),
            target_first_token=g,competitor_token=int(order[1] if success else order[0]),source_span=[1,2]))
    with open(out/'cohort.jsonl','w') as f:
        for r in rows: f.write(json.dumps(r)+'\n')
    write_json(out/'decoys.json',{f'relation{i}':list(range(1,32)) for i in range(2)})
    write_json(out/'WARNING.json',dict(purpose='Random model and fabricated task targets for software smoke testing only'))


if __name__=='__main__': main()

"""Index the included compact results without loading model weights."""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for group in sorted((ROOT/'results/observed').iterdir()):
    print('\n'+group.name)
    for p in sorted(group.rglob('*.csv')):
        with p.open(newline='') as f:
            reader=csv.reader(f);columns=next(reader,[]);n=sum(1 for _ in reader)
        print(f'  {p.relative_to(group)}: {n} rows, {len(columns)} columns')

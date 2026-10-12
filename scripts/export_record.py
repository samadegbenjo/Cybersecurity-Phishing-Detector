"""Export one existing offline CSV record, WITHOUT its label, for API/popup checks."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phishing.schema import VERSION,NAMES,load_csv
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--dataset',default='dataset.csv'); p.add_argument('--row',type=int,default=0); p.add_argument('--output',default='artifacts/record.json')
a=p.parse_args(); df=load_csv(a.dataset)
features={n:None if pd.isna(df.iloc[a.row][n]) else float(df.iloc[a.row][n]) for n in NAMES}
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text(json.dumps({'schema_version':VERSION,'features':features},indent=2)+'\n')

"""Training/evaluation CLI. See README for the frozen evaluation protocol."""
import argparse
import json
from pathlib import Path
from phishing.research import train, evaluate, audit

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command', choices=['train','evaluate','audit'], nargs='?', default='train')
    p.add_argument('--dataset', default='dataset.csv')
    p.add_argument('--output', default='artifacts')
    p.add_argument('--mode',choices=['offline','live_url'],default='offline')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--precision-target', type=float, default=.95)
    args=p.parse_args()
    if args.command=='train':
        result=train(args.dataset,args.output,args.seed,args.precision_target,args.mode)
        print(json.dumps({'selected_model':result['selected_model'],'test':result['test']},indent=2))
    else:
        print(json.dumps(audit(args.dataset) if args.command=='audit' else evaluate(args.dataset,args.output),indent=2))

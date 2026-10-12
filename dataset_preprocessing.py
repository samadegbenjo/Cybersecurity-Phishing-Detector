"""Legacy entry point replaced with audit only: never preprocess the full dataset."""
import json
from phishing.research import audit
if __name__ == '__main__':
    print(json.dumps(audit('dataset.csv'),indent=2))
    print('To train with split-before-fit preprocessing: python model_training.py train')

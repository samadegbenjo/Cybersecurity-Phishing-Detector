"""Reproducible raw-data split, training-only preprocessing, validation selection."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .schema import NAMES, SCHEMA, VERSION, load_csv, normalize_missing


def audit(path):
    df = load_csv(path)
    x = df[NAMES]
    conflicts = df.groupby(NAMES, dropna=False).phishing.nunique()
    return dict(dataset_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                rows=len(df), features=len(NAMES), class_counts=counts(df.phishing),
                missing=x.isna().sum().astype(int).to_dict(),
                minus_one=x.eq(-1).sum().astype(int).to_dict(),
                duplicate_rows=int(df.duplicated().sum()),
                conflicting_feature_vectors=int((conflicts > 1).sum()),
                constants=x.columns[x.nunique(dropna=False)==1].tolist(),
                identifiers=[], limitation='No URLs/domains/timestamps: domain and temporal isolation cannot be verified.')


def counts(y):
    return {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()}


def split_data(df, seed=42):
    # Remove ALL conflicting vectors instead of choosing one label, then deduplicate.
    conflict = df.groupby(NAMES, dropna=False).phishing.transform('nunique') > 1
    clean = df.loc[~conflict].drop_duplicates(subset=NAMES).copy()
    train, rest = train_test_split(clean.index, test_size=.4, stratify=clean.phishing, random_state=seed)
    val, test = train_test_split(rest, test_size=.5, stratify=clean.loc[rest, 'phishing'], random_state=seed)
    return clean, {'train': sorted(train.tolist()), 'validation': sorted(val.tolist()), 'test': sorted(test.tolist())}


def make_pipeline(name, seed=42):
    steps = [('imputer', SimpleImputer(strategy='median', add_indicator=True))]
    if name == 'logistic_regression':
        steps += [('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=3000, random_state=seed))]
    elif name == 'random_forest':
        steps += [('classifier', RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=2))]
    else:
        raise ValueError('unknown model')
    return Pipeline(steps)


def phishing_probability(model, x):
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError('model has no phishing class 1')
    return model.predict_proba(x)[:, classes.index(1)]


def metrics(y, score, threshold):
    pred = (np.asarray(score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)),
                accuracy=float(accuracy_score(y, pred)),
                pr_auc_average_precision=float(average_precision_score(y, score)) if len(set(y))==2 else None,
                roc_auc=float(roc_auc_score(y, score)) if len(set(y))==2 else None,
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
                false_positive_rate=float(fp/(tn+fp)) if tn+fp else None,
                class_counts=counts(y), threshold=float(threshold))


def choose_threshold(y, score, target=.95):
    precision, recall, thresholds = precision_recall_curve(y, score)
    eligible = np.flatnonzero((precision[:-1] >= target) & (recall[:-1] > 0))
    if not len(eligible):
        return .5, False  # Explicit fallback; never count a no-positive prediction as success.
    best = max(eligible, key=lambda i: (recall[i], precision[i], thresholds[i]))
    return float(thresholds[best]), True


def train(path, output, seed=42, target=.95):
    if not 0 < target <= 1:
        raise ValueError('precision target must be in (0,1]')
    df = load_csv(path)
    clean, splits = split_data(df, seed)
    x = normalize_missing(clean[NAMES])
    if x.loc[splits['train']].isna().all().any():
        raise ValueError('training contains an all-missing feature; revise schema rather than fill arbitrary zeros')
    candidates, fitted = {}, {}
    for name in ['logistic_regression', 'random_forest']:
        model = make_pipeline(name, seed)
        model.fit(x.loc[splits['train']], clean.loc[splits['train'], 'phishing'])
        score = phishing_probability(model, x.loc[splits['validation']])
        threshold, achieved = choose_threshold(clean.loc[splits['validation'], 'phishing'], score, target)
        candidates[name] = dict(validation=metrics(clean.loc[splits['validation'], 'phishing'], score, threshold),
                                precision_target_achieved=achieved, parameters=model.get_params()['classifier'].get_params())
        fitted[name] = model
    # Frozen choice using validation only. Test data touched below once for selected candidate.
    selected = max(candidates, key=lambda n: (candidates[n]['precision_target_achieved'],
                   candidates[n]['validation']['recall'] if candidates[n]['precision_target_achieved'] else candidates[n]['validation']['f1'],
                   candidates[n]['validation']['pr_auc_average_precision']))
    model = fitted[selected]
    threshold = candidates[selected]['validation']['threshold']
    report = dict(audit=audit(path), schema_version=VERSION, seed=seed, precision_target=target,
                  threshold_objective='maximize validation recall subject to precision >= target; fallback 0.5 if infeasible',
                  selection='validation target achieved, then recall (fallback F1), then average precision',
                  candidates=candidates, selected_model=selected,
                  precision_target_achieved=candidates[selected]['precision_target_achieved'],
                  test=metrics(clean.loc[splits['test'], 'phishing'], phishing_probability(model, x.loc[splits['test']]), threshold),
                  splits={k:dict(rows=len(v), class_counts=counts(clean.loc[v,'phishing'])) for k,v in splits.items()},
                  excluded_rows=len(df)-len(clean), python=platform.python_version(),
                  dependencies={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','scipy','joblib','Flask']},
                  pca='not used: no demonstrated benefit; RF does not require scaling',
                  scope='offline feature records only; no verified browser equivalence')
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    manifest=dict(dataset_sha256=report['audit']['dataset_sha256'], seed=seed, row_indices=splits,
                  rule='remove conflicting vectors, deduplicate features, stratified 60/20/20; row indices refer to original CSV')
    (out/'splits.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report['split_manifest_sha256']=hashlib.sha256((out/'splits.json').read_bytes()).hexdigest()
    joblib.dump(dict(pipeline=model, threshold=threshold, schema=SCHEMA, metadata=report), out/'model.joblib')
    report['model_sha256']=hashlib.sha256((out/'model.joblib').read_bytes()).hexdigest()
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def evaluate(path, output):
    """Re-evaluate ONLY the frozen test indices; no fitting or threshold search."""
    out=Path(output)
    report=json.loads((out/'metrics.json').read_text())
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=report['audit']['dataset_sha256']:
        raise ValueError('dataset checksum differs')
    if hashlib.sha256((out/'splits.json').read_bytes()).hexdigest()!=report['split_manifest_sha256']:
        raise ValueError('split manifest checksum differs')
    if hashlib.sha256((out/'model.joblib').read_bytes()).hexdigest()!=report['model_sha256']:
        raise ValueError('model checksum differs')
    artifact=joblib.load(out/'model.joblib')  # Trust only artifacts you trained yourself.
    df=load_csv(path); ids=json.loads((out/'splits.json').read_text())['row_indices']['test']
    return metrics(df.loc[ids,'phishing'], phishing_probability(artifact['pipeline'],normalize_missing(df.loc[ids,NAMES])), artifact['threshold'])

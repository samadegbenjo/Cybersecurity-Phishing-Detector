import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from phishing.schema import NAMES,VERSION,load_csv,normalize_missing,validate_features
from phishing.research import (train,evaluate,make_pipeline,split_data,phishing_probability,choose_threshold,metrics)
from app import create_app

FIXTURE=Path(__file__).parent/'fixtures/synthetic.csv'

@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    out=tmp_path_factory.mktemp('artifact')
    report=train(FIXTURE,out)
    return out,report

@pytest.fixture
def record():
    row=load_csv(FIXTURE).iloc[0]
    return {n:float(row[n]) for n in NAMES}


def test_named_order_and_missing(record):
    expected=validate_features(record)
    assert expected.equals(validate_features(dict(reversed(list(record.items())))))
    record['time_response']=None
    record['time_domain_activation']=-1
    record['qty_dot_directory']=-1
    x=validate_features(record)
    assert np.isnan(x.time_response[0]) and np.isnan(x.time_domain_activation[0])
    assert x.qty_dot_directory[0]==-1

@pytest.mark.parametrize('bad',[True,'3',float('inf'),float('nan'),.1,-2,10**400])
def test_invalid_values(record,bad):
    record[NAMES[0]]=bad
    with pytest.raises(ValueError): validate_features(record)

def test_missing_count(record):
    with pytest.raises(ValueError): validate_features(list(record.values()))
    record.pop(NAMES[0])
    with pytest.raises(ValueError): validate_features(record)
    with pytest.raises(ValueError): validate_features({n:None for n in NAMES})


def test_class_probability_mapping():
    class Reversed:
        classes_=np.array([1,0])
        def predict_proba(self,x): return np.array([[.8,.2]])
    assert phishing_probability(Reversed(),None)[0]==.8


def test_threshold_validation_and_boundary():
    threshold,ok=choose_threshold([0,0,1,1],[.05,.3,.8,.9],.95)
    assert ok and threshold==.8
    m=metrics([0,1],[threshold-.001,threshold],threshold)
    assert m['confusion_matrix']==[[1,0],[0,1]]
    assert choose_threshold([1,0],[.1,.9],1)==(.5,False)
    assert metrics([0,0],[.1,.2],.5)['roc_auc'] is None


def test_raw_split_deduplicates_and_excludes_conflicts():
    df=load_csv(FIXTURE)
    dup=df.iloc[[0]].copy(); conflict=df.iloc[[1]].copy(); conflict.phishing=1-conflict.phishing
    df=pd.concat([df,dup,conflict],ignore_index=True)
    clean,splits=split_data(df)
    assert 1 not in clean.index and 121 not in clean.index and 120 not in clean.index
    sets=[set(v) for v in splits.values()]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
    assert set.union(*sets)==set(clean.index)


def test_train_only_preprocessing_and_validation_selection(monkeypatch,tmp_path):
    import phishing.research as research
    df=load_csv(FIXTURE); clean,splits=split_data(df)
    expected=set(splits['train']); calls=[]; threshold_rows=[]
    original_fit=research.Pipeline.fit
    def spy_fit(self,x,y,*args,**kwargs):
        calls.append(set(x.index)); assert set(x.index)==expected
        return original_fit(self,x,y,*args,**kwargs)
    original_choose=research.choose_threshold
    def spy_threshold(y,scores,target):
        threshold_rows.append(set(y.index)); assert set(y.index)==set(splits['validation'])
        return original_choose(y,scores,target)
    monkeypatch.setattr(research.Pipeline,'fit',spy_fit)
    monkeypatch.setattr(research,'choose_threshold',spy_threshold)
    research.train(FIXTURE,tmp_path)
    assert len(calls)==2 and len(threshold_rows)==2
    import joblib
    fitted=joblib.load(tmp_path/'model.joblib')['pipeline']
    x=normalize_missing(clean[NAMES]); expected_medians=x.loc[splits['train']].median().values
    np.testing.assert_allclose(fitted.named_steps['imputer'].statistics_,expected_medians)
    if 'scaler' in fitted.named_steps:
        transformed=fitted.named_steps['imputer'].transform(x.loc[splits['train']])
        np.testing.assert_allclose(fitted.named_steps['scaler'].mean_,transformed.mean(axis=0))


def test_evaluation_is_frozen(trained):
    out,report=trained
    assert evaluate(FIXTURE,out)==report['test']
    with pytest.raises(ValueError): evaluate(Path('dataset.csv'),out)


def test_api_schema_errors_mapping_and_metadata(trained,record):
    out,_=trained
    client=create_app(out/'model.joblib').test_client()
    assert client.get('/health').json['model_ready']
    assert client.get('/metadata').json['schema']['version']==VERSION
    body={'features':record,'schema_version':VERSION}
    response=client.post('/predict',json=body)
    assert response.status_code==200
    data=response.json
    import joblib
    artifact=joblib.load(out/'model.joblib')
    expected=float(phishing_probability(artifact['pipeline'],validate_features(record))[0])
    assert data['risk_score']==pytest.approx(expected,abs=1e-12)
    assert data['phishing']==(data['risk_score']>=data['threshold'])
    assert data['state']==('suspicious' if data['phishing'] else 'below_threshold')
    for bad in [[],{},dict(body,schema_version='unknown'),dict(body,features=[1,2]),dict(body,features={})]:
        r=client.post('/predict',json=bad); assert r.status_code==422 and 'phishing' not in r.json
    record['time_response']=None
    assert client.post('/predict',json=body).json['missing_feature_count']==1
    assert client.post('/predict',data='{',content_type='application/json').status_code==400
    assert client.post('/predict',data='hi',content_type='text/plain').status_code==415


def test_unavailable_model_and_origin(trained,record,tmp_path):
    client=create_app(tmp_path/'absent').test_client()
    r=client.post('/predict',json={'features':record,'schema_version':VERSION})
    assert r.status_code==503 and r.json['state']=='unable' and 'phishing' not in r.json
    origin='chrome-extension://'+'a'*32
    client=create_app(trained[0]/'model.joblib',[origin]).test_client()
    assert client.post('/observe',headers={'Origin':'https://untrusted.example'},json={}).status_code==403
    response=client.options('/predict',headers={'Origin':origin,'Access-Control-Request-Method':'POST'})
    assert response.status_code==204 and response.headers['Access-Control-Allow-Origin']==origin


def test_browser_demo_never_returns_verdict(trained):
    client=create_app(trained[0]/'model.joblib').test_client()
    for url in ['http://127.0.0.1:8000/page','https://example.invalid/?q=<script>']:
        r=client.post('/observe',json={'schema_version':'url-observation-v1','url':url})
        assert r.status_code==200 and r.json['risk_score'] is None and r.json['state']=='unable'
        assert 'phishing' not in r.json and r.json['observations']['url_length']==len(url)
    assert client.post('/observe',json={'schema_version':'url-observation-v1','url':'chrome://extensions/'}).status_code==422


def test_scaler_statistics_exclude_validation_and_test():
    df=load_csv(FIXTURE); clean,splits=split_data(df)
    x=normalize_missing(clean[NAMES])
    pipeline=make_pipeline('logistic_regression')
    pipeline.fit(x.loc[splits['train']],clean.loc[splits['train'],'phishing'])
    imputed=pipeline.named_steps['imputer'].transform(x.loc[splits['train']])
    np.testing.assert_allclose(pipeline.named_steps['scaler'].mean_,imputed.mean(axis=0))
    before=pipeline.named_steps['scaler'].mean_.copy()
    holdout=x.loc[splits['test']].copy(); holdout['length_url']=10**9
    pipeline.predict_proba(holdout)
    np.testing.assert_array_equal(pipeline.named_steps['scaler'].mean_,before)


def test_prediction_exception_is_error(trained,record,monkeypatch):
    import app
    def fail(*args): raise RuntimeError('test failure')
    monkeypatch.setattr(app,'phishing_probability',fail)
    r=app.create_app(trained[0]/'model.joblib').test_client().post('/predict',json={'features':record,'schema_version':VERSION})
    assert r.status_code==500 and r.json['state']=='unable' and 'phishing' not in r.json

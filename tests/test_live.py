import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from app import create_app
from phishing.live import LIVE_NAMES,LIVE_SCHEMA,LIVE_VERSION,extract_url
from phishing.research import train,evaluate,phishing_probability
from phishing.schema import load_csv

FIXTURE=Path(__file__).parent/'fixtures/synthetic.csv'

@pytest.fixture(scope='module')
def live_artifact(tmp_path_factory):
    out=tmp_path_factory.mktemp('live')
    report=train(FIXTURE,out,mode='live_url')
    return out,report


def test_upstream_reference_parity():
    reference=json.loads((FIXTURE.parent/'lexical_reference.json').read_text())
    assert reference['upstream_commit']==LIVE_SCHEMA['evidence']['commit']
    for case in reference['cases']:
        actual=extract_url(case['url'])
        assert list(actual.columns)==LIVE_NAMES
        assert actual.iloc[0].to_dict()==case['features']
        assert actual.shape==(1,39) and not actual.isna().any().any()


def test_parser_preserves_upstream_separator_and_host_semantics():
    x=extract_url('https://alice@client-server.invalid:8443/a;b?x=1&y=2#part').iloc[0]
    joined='alice@client-server.invalid:8443/abx=1&y=2part'
    assert x.length_url==len(joined) and x.qty_questionmark_url==0 and x.qty_hashtag_url==0
    assert x.qty_at_domain==1 and x.server_client_domain==1
    assert x.domain_length==len('alice@client-server.invalid:8443')
    assert extract_url('http://192.0.2.1/').domain_in_ip[0]==1
    assert extract_url('http://192.0.2.1:8080/').domain_in_ip[0]==0
    assert not extract_url('https://example.invalid/').equals(extract_url('https://client-server.invalid/a-b.c'))

@pytest.mark.parametrize('url',[None,1,'chrome://extensions','file:///test','javascript:alert(1)',
                                'http:///path','https://example.invalid:bad/','https://[broken/',
                                'https://example.invalid/\n','https://example.invalid/é','https://example.invalid/a\\b','x'*9000])
def test_unsupported_urls_fail(url):
    with pytest.raises(ValueError):extract_url(url)


def test_live_train_groups_identical_vectors_even_with_contradictory_labels(tmp_path):
    df=load_csv(FIXTURE)
    a=df.iloc[[0]].copy(); a.time_response=99; a.phishing=1-a.phishing
    df=pd.concat([df,a],ignore_index=True)
    path=tmp_path/'synthetic_collision.csv';df.to_csv(path,index=False)
    report=train(path,tmp_path/'model',mode='live_url')
    ids=json.loads((tmp_path/'model/splits.json').read_text())['row_indices']
    groups=[set(pd.util.hash_pandas_object(df.loc[rows,LIVE_NAMES],index=False)) for rows in ids.values()]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i+1,3))
    assert sum(0 in rows and 120 in rows for rows in ids.values())==1
    assert report['lexical_colliding_label_groups']==1


def test_live_prediction_matches_saved_pipeline_and_retains_offline(live_artifact,tmp_path):
    out,report=live_artifact
    client=create_app(tmp_path/'no_offline',live_model_path=out/'model.joblib').test_client()
    assert client.get('/health').json['live_model_ready']
    assert not client.get('/health').json['model_ready']
    body={'schema_version':LIVE_VERSION,'url':'http://127.0.0.1:8000/controlled.html?check=1'}
    r=client.post('/predict-url',json=body)
    assert r.status_code==200
    data=r.json; artifact=joblib.load(out/'model.joblib')
    expected=phishing_probability(artifact['pipeline'],extract_url(body['url']))[0]
    assert data['risk_score']==pytest.approx(expected)
    assert data['threshold']==report['test']['threshold'] and data['feature_count']==39
    assert data['phishing']==(expected>=data['threshold']) and data['mode']=='live_url'
    assert client.get('/metadata').json['live']['schema']==LIVE_SCHEMA
    assert evaluate(FIXTURE,out)==report['test']
    for bad in [{},dict(body,schema_version='offline-v1'),dict(body,features=[0]*39),dict(body,url='chrome://extensions')]:
        r=client.post('/predict-url',json=bad)
        assert r.status_code==422 and r.json['state']=='unable' and 'phishing' not in r.json


def test_live_missing_model_and_prediction_failure(live_artifact,tmp_path,monkeypatch):
    body={'schema_version':LIVE_VERSION,'url':'https://example.invalid/'}
    client=create_app(tmp_path/'none',live_model_path=tmp_path/'missing').test_client()
    r=client.post('/predict-url',json=body)
    assert r.status_code==503 and 'phishing' not in r.json
    import app
    def fail(*args): raise RuntimeError('synthetic error')
    monkeypatch.setattr(app,'phishing_probability',fail)
    client=create_app(tmp_path/'none',live_model_path=live_artifact[0]/'model.joblib').test_client()
    r=client.post('/predict-url',json=body)
    assert r.status_code==500 and r.json['state']=='unable' and 'phishing' not in r.json

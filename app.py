"""Loopback-only offline inference and URL-observation demonstration API."""
import argparse
import json
import logging
from pathlib import Path
from urllib.parse import urlsplit
import joblib
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
from phishing.schema import SCHEMA, VERSION, NAMES, validate_features
from phishing.research import phishing_probability

LIMITATION = 'Offline research model; live URL extraction equivalence is unverified. Scores are uncalibrated and do not guarantee safety.'


def create_app(model_path='artifacts/model.joblib', allowed_origins=()):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 64 * 1024
    artifact = None
    if Path(model_path).exists():
        try:
            candidate = joblib.load(model_path)  # Never load untrusted pickle/joblib files.
            if candidate['schema'] != SCHEMA or not 0 <= candidate['threshold'] <= 1:
                raise ValueError('artifact schema or threshold mismatch')
            if list(candidate['pipeline'].feature_names_in_) != NAMES:
                raise ValueError('artifact feature ordering mismatch')
            if set(candidate['pipeline'].classes_) != {0,1}:
                raise ValueError('artifact class mapping mismatch')
            artifact = candidate
        except Exception:
            logging.exception('Model could not be loaded; inference disabled')
    allowed = set(allowed_origins)

    @app.before_request
    def origin_check():
        origin = request.headers.get('Origin')
        if origin and origin not in allowed:
            return jsonify(error='origin_not_allowed', state='unable'), 403
        if request.method == 'OPTIONS':
            return '', 204

    @app.after_request
    def cors(response):
        origin = request.headers.get('Origin')
        if origin in allowed:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Vary'] = 'Origin'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(state='unable', error=exc.name), exc.code

    @app.get('/health')
    def health():
        return jsonify(status='ok', model_ready=artifact is not None, mode='offline', schema_version=VERSION)

    @app.get('/metadata')
    def metadata():
        return jsonify(schema=SCHEMA, model_ready=artifact is not None,
                       model=artifact['metadata']['selected_model'] if artifact else None,
                       threshold=artifact['threshold'] if artifact else None,
                       limitation=LIMITATION)

    @app.post('/predict')
    def predict():
        body=request.get_json()
        try:
            if not isinstance(body,dict) or set(body)!={'schema_version','features'}:
                raise ValueError('request requires only schema_version and features')
            if body['schema_version'] != VERSION:
                raise ValueError('unsupported schema_version')
            x=validate_features(body['features'])
        except (ValueError,TypeError) as exc:
            return jsonify(state='unable', error=str(exc)),422
        if artifact is None:
            return jsonify(state='unable',error='model_unavailable; run training first'),503
        try:
            score=float(phishing_probability(artifact['pipeline'],x)[0])
            if not 0 <= score <= 1:
                raise ValueError('invalid model score')
            suspicious=score>=artifact['threshold']
            return jsonify(state='suspicious' if suspicious else 'below_threshold',
                           phishing=suspicious, risk_score=score,threshold=artifact['threshold'],
                           schema_version=VERSION,mode='offline',limitation=LIMITATION,
                           missing_feature_count=int(x.isna().sum().sum()))
        except Exception:
            app.logger.exception('Prediction failed')
            return jsonify(state='unable',error='prediction_failed'),500

    @app.post('/observe')
    def observe():
        body=request.get_json()
        if not isinstance(body,dict) or set(body)!={'schema_version','url'} or body['schema_version']!='url-observation-v1':
            return jsonify(state='unable',error='unsupported observation request'),422
        url=body['url']
        if not isinstance(url,str) or len(url)>8192:
            return jsonify(state='unable',error='invalid URL'),422
        try:
            parsed=urlsplit(url)
            if parsed.scheme not in ('http','https') or not parsed.hostname:
                raise ValueError('unsupported page')
        except ValueError:
            return jsonify(state='unable',error='unsupported page'),422
        # Do not fetch URLs, access DNS, or feed partial observations to the offline model.
        return jsonify(state='unable',mode='browser_demo',schema_version='url-observation-v1',
                       reason='No verified live feature contract; this page cannot be classified.',
                       risk_score=None,threshold=artifact['threshold'] if artifact else None,
                       observations=dict(url_length=len(url),dot_count=url.count('.'),
                                         at_count=url.count('@'),uses_https=parsed.scheme=='https'),
                       limitation=LIMITATION)
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='artifacts/model.joblib')
    parser.add_argument('--extension-id',help='ID from chrome://extensions; enables exact-origin CORS')
    args=parser.parse_args()
    origins=() if args.extension_id is None else ('chrome-extension://'+args.extension_id,)
    create_app(args.model,origins).run(host='127.0.0.1',port=5000,debug=False)

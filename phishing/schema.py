"""Single ordered feature contract for CSV training and API inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

SCHEMA = json.loads(Path(__file__).with_name('feature_schema.json').read_text())
VERSION = SCHEMA['version']
NAMES = [f['name'] for f in SCHEMA['features']]


def validate_features(features):
    if not isinstance(features, dict) or set(features) != set(NAMES):
        raise ValueError(f'features must contain exactly the {len(NAMES)} named schema fields')
    values = []
    for field in SCHEMA['features']:
        v = features[field['name']]
        if v is None:
            values.append(np.nan)
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{field['name']}: expected numeric value or null")
        try:
            finite = np.isfinite(float(v))
        except (OverflowError, ValueError):
            finite = False
        if not finite or abs(v) > 1e12:
            raise ValueError(f"{field['name']}: nonfinite or out-of-range value")
        if field['type'] == 'integer' and v != int(v):
            raise ValueError(f"{field['name']}: expected integer")
        if v < 0 and not (v == -1 and field['minus_one'] != 'not allowed'):
            raise ValueError(f"{field['name']}: unsupported negative value")
        values.append(float(v))
    frame = pd.DataFrame([values], columns=NAMES)
    frame = normalize_missing(frame)
    if frame.isna().all(axis=1).any():
        raise ValueError('all features are missing')
    return frame


def normalize_missing(frame):
    """Deterministic sentinel handling, not a fitted transformation."""
    result = frame[NAMES].copy()
    for f in SCHEMA['features']:
        if f['minus_one'].startswith('convert'):
            result[f['name']] = result[f['name']].replace(-1, np.nan)
    return result


def load_csv(path):
    df = pd.read_csv(path)
    if list(df.columns) != NAMES + ['phishing']:
        raise ValueError('CSV column order/schema does not match offline-v1')
    if df.phishing.isna().any() or not set(df.phishing.unique()) <= {0, 1}:
        raise ValueError('phishing labels must be 0 or 1')
    for f in SCHEMA['features']:
        s = df[f['name']]
        if not pd.api.types.is_numeric_dtype(s) or np.isinf(s).any():
            raise ValueError(f"{f['name']}: nonnumeric or infinite CSV values")
        valid = s.dropna()
        if (valid.abs() > 1e12).any() or (f['type'] == 'integer' and (valid % 1 != 0).any()):
            raise ValueError(f"{f['name']}: invalid numeric range/type")
        negatives = valid[valid < 0]
        if len(negatives) and (f['minus_one'] == 'not allowed' or (negatives != -1).any()):
            raise ValueError(f"{f['name']}: unsupported negative values")
    return df

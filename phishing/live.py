"""Network-free active-URL contract aligned to a verified upstream lexical subset."""
import json
import ipaddress
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd

LIVE_SCHEMA=json.loads(Path(__file__).with_name('live_schema.json').read_text())
LIVE_NAMES=[f['name'] for f in LIVE_SCHEMA['features']]
LIVE_VERSION=LIVE_SCHEMA['version']
SYMBOLS={"dot":".","hyphen":"-","underline":"_","slash":"/","questionmark":"?","equal":"=","at":"@","and":"&","exclamation":"!","space":" ","tilde":"~","comma":",","plus":"+","asterisk":"*","hashtag":"#","dollar":"$","percent":"%"}
LIVE_LIMITATION='Experimental URL-only classifier; page content is not inspected. Historical extraction parity and generalization remain limited. Score is uncalibrated; below threshold does not guarantee safety.'


def extract_url(url):
    if not isinstance(url,str) or not 1 <= len(url) <= 8192:
        raise ValueError('URL must be a string of 1–8192 characters')
    # Browser URLs are encoded ASCII; reject unsupported Unicode/control inputs rather than guess normalization.
    if not url.isascii() or any(ord(c)<=32 or ord(c)==127 for c in url) or '\\' in url:
        raise ValueError('URL must be ASCII-encoded without whitespace, controls or backslashes')
    try:
        parsed=urlparse(url)
        if parsed.scheme not in ('http','https') or not parsed.hostname:
            raise ValueError('Only HTTP(S) pages are supported')
        parsed.port  # Validate invalid port forms even though netloc is counted verbatim.
    except ValueError as exc:
        raise ValueError('Unsupported or malformed URL') from exc
    # Precisely reproduce author's start_url representation: no scheme or ?, ;, # separators.
    lexical=parsed.netloc+parsed.path+parsed.params+parsed.query+parsed.fragment
    host=parsed.netloc  # Deliberately includes userinfo and port, as in the upstream implementation.
    features={}
    for field in LIVE_SCHEMA['features']:
        name=field['name']
        if name.startswith('qty_') and name not in ('qty_vowels_domain',):
            _,token,component=name.split('_')
            features[name]=(lexical if component=='url' else host).count(SYMBOLS[token])
        elif name=='length_url': features[name]=len(lexical)
        elif name=='domain_length': features[name]=len(host)
        elif name=='qty_vowels_domain': features[name]=sum(host.lower().count(c) for c in 'aeiou')
        elif name=='server_client_domain': features[name]=int('server' in host.lower() or 'client' in host.lower())
        elif name=='domain_in_ip':
            try: ipaddress.ip_address(host); features[name]=1
            except ValueError: features[name]=0
        else: raise ValueError('Unsupported live feature in schema')
    return pd.DataFrame([features],columns=LIVE_NAMES)

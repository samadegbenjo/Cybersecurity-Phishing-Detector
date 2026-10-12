"""Generate controlled parity fixture from reviewed upstream pure functions, without network requests.
Usage: python scripts/verify_lexical_reference.py /path/to/url-feature-extractor
The upstream repository is not vendored; only generated factual feature values are saved.
"""
import ast
import hashlib
import ipaddress
import json
from pathlib import Path
import subprocess
import sys
from urllib import parse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phishing.live import LIVE_NAMES,SYMBOLS,extract_url

root=Path(sys.argv[1]); path=root/'lib/functions.py'; source=path.read_text()
functions={'start_url','count','count_vowels','length','valid_ip','check_word_server_client'}
tree=ast.parse(source); nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions]
if {n.name for n in nodes}!=functions: raise ValueError('Upstream pure-function set missing')
namespace={'parse':parse,'ipaddress':ipaddress}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
urls=['https://example.invalid/','http://127.0.0.1:8000/controlled.html?check=1',
      'https://alice:secret@client-server.invalid:8443/a;b?x=1&y=%23#fragment',
      'https://example.invalid/a//b.c?q=a?b#part','https://[::1]/a',
      'https://example.invalid/%E2%9C%93?blank=&encoded=%40', 'http://192.0.2.1/a',
      'https://EXAMPLE.invalid/~a_!,$+*%-/#thing']
cases=[]
for url in urls:
 parts=namespace['start_url'](url); result={}
 for name in LIVE_NAMES:
  if name.startswith('qty_') and name!='qty_vowels_domain':
   _,token,part=name.split('_');result[name]=namespace['count'](parts['url' if part=='url' else 'host'],SYMBOLS[token])
  elif name=='length_url':result[name]=namespace['length'](parts['url'])
  elif name=='domain_length':result[name]=namespace['length'](parts['host'])
  elif name=='qty_vowels_domain':result[name]=namespace['count_vowels'](parts['host'])
  elif name=='domain_in_ip':result[name]=int(namespace['valid_ip'](parts['host']))
  elif name=='server_client_domain':result[name]=int(namespace['check_word_server_client'](parts['host']))
 if extract_url(url).iloc[0].to_dict()!=result: raise ValueError('Reference parity failed')
 cases.append(dict(url=url,features=result))
report=dict(synthetic=True,source='https://github.com/gregavrbancic/url-feature-extractor',
            upstream_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cases=cases)
Path('tests/fixtures/lexical_reference.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'{len(cases)} controlled URLs x {len(LIVE_NAMES)} fields: upstream parity verified; no sites visited')

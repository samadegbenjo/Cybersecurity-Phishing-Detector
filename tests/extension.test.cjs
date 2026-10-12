const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const client=require('../extension/client.js');
const good={mode:'offline',schema_version:'offline-v1',state:'suspicious',phishing:true,risk_score:.8,threshold:.7};
const mock=data=>async()=>({ok:true,json:async()=>data});
test('threshold and benign/suspicious/error states are distinct',async()=>{
  assert.equal((await client.evaluateRecord({},mock(good))).state,'suspicious');
  assert.equal((await client.evaluateRecord({},mock({...good,risk_score:.4,phishing:false,state:'below_threshold'}))).state,'below_threshold');
  assert.equal((await client.evaluateRecord({},mock({...good,risk_score:.4}))).state,'unable');
});
test('failed requests, malformed JSON, HTTP failures, and invalid responses fail closed',async()=>{
  for(const fetcher of [async()=>{throw Error('offline')},async()=>({ok:false,status:503}),mock({phishing:false}),async()=>({ok:true,json:async()=>{throw Error('bad json')}})]){
    const r=await client.evaluateRecord({},fetcher);
    assert.equal(r.state,'unable'); assert.match(r.text,/Unable to evaluate/);
  }
});
test('timeout aborts request',async()=>{
  const hang=(_url,{signal})=>new Promise((resolve,reject)=>signal.addEventListener('abort',()=>reject(Error('timeout'))));
  assert.equal((await client.evaluateRecord({},hang,10)).state,'unable');
});
test('unsupported page never contacts backend',async()=>{
  let calls=0; const fetcher=async()=>{calls++;throw Error('unexpected')};
  const r=await client.observePage('chrome://extensions',fetcher);assert.equal(r.state,'unable');assert.equal(calls,0);
});
test('popup renders external strings using textContent',()=>{
  const elements={}; const node=()=>({dataset:{},children:[],addEventListener(){},replaceChildren(){this.children=[]},appendChild(x){this.children.push(x)}});
  for(const id of ['result','record','evaluate']) elements[id]=node();
  const context={document:{getElementById:id=>elements[id],createElement:()=>node()},PhishingClient:client};
  vm.createContext(context);vm.runInContext(fs.readFileSync('extension/popup.js','utf8'),context);
  vm.runInContext(`show({state:'unable',text:'<img src=x onerror=alert(1)>'})`,context);
  assert.equal(elements.result.children[0].textContent,'<img src=x onerror=alert(1)>');
  assert.equal(elements.result.innerHTML,undefined);
});
test('MV3 minimal permissions',()=>{
  const m=JSON.parse(fs.readFileSync('extension/manifest.json'));
  assert.equal(m.manifest_version,3);assert.deepEqual(m.permissions,['activeTab']);
  assert.deepEqual(m.host_permissions,['http://127.0.0.1/*']);assert.equal(m.content_scripts,undefined);
});
test('real loopback API to model to popup client; optional standalone server check',async t=>{
  if(!process.env.PHISHING_INTEGRATION){t.skip('Set PHISHING_INTEGRATION=1 with server on port 5000');return;}
  const record=JSON.parse(fs.readFileSync('artifacts/record.json'));
  const result=await client.evaluateRecord(record);
  assert.ok(['suspicious','below_threshold'].includes(result.state));
  const observation=await client.observePage('http://127.0.0.1:8000/controlled');
  assert.equal(observation.state,'unable');assert.equal(observation.observations.url_length,'http://127.0.0.1:8000/controlled'.length);
});

test('active page uses actual tab URL and rejects malformed live responses',async()=>{
  const url='https://example.invalid/a?x=1';let captured;
  const data={...good,mode:'live_url',schema_version:'live-url-v1',feature_count:39,limitation:'URL only'};
  const fetcher=async(endpoint,options)=>{captured={endpoint,body:JSON.parse(options.body)};return {ok:true,json:async()=>data}};
  const result=await client.evaluatePage(url,fetcher);
  assert.equal(result.state,'suspicious');assert.equal(result.evaluated_url,url);
  assert.equal(captured.endpoint,'http://127.0.0.1:5000/predict-url');
  assert.deepEqual(captured.body,{schema_version:'live-url-v1',url});
  for(const invalid of [{...data,feature_count:111},{...data,schema_version:'offline-v1'},{...data,risk_score:NaN},{...data,threshold:.9}]){
    assert.equal((await client.evaluatePage(url,mock(invalid))).state,'unable');
  }
  assert.equal((await client.evaluatePage(url,async()=>{throw Error('offline')})).state,'unable');
});
test('active-page click flows through tabs.query to live client to safe text display',async()=>{
  const elements={}; const node=()=>({dataset:{},children:[],callbacks:{},disabled:false,addEventListener(k,v){this.callbacks[k]=v},replaceChildren(){this.children=[]},appendChild(x){this.children.push(x)}});
  for(const id of ['result','record','evaluate']) elements[id]=node();
  const url='https://example.invalid/?x=%3Cscript%3E';let received;
  const context={document:{getElementById:id=>elements[id],createElement:()=>node()},
    chrome:{tabs:{query:async()=>[{id:1,url}]}},PhishingClient:{evaluatePage:async u=>{received=u;return {state:'below_threshold',text:'Active URL: below threshold',risk_score:.2,threshold:.7,evaluated_url:u}}}};
  vm.createContext(context);vm.runInContext(fs.readFileSync('extension/popup.js','utf8'),context);
  await elements.evaluate.callbacks.click({target:elements.evaluate});
  assert.equal(received,url);assert.equal(elements.result.dataset.state,'below_threshold');
  assert.ok(elements.result.children.some(x=>x.textContent==='URL evaluated: '+url));
  assert.equal(elements.evaluate.disabled,false);
});
test('real active URL inference through loopback',async t=>{
  if(!process.env.PHISHING_INTEGRATION){t.skip('Run check_integration.py');return;}
  const result=await client.evaluatePage('http://127.0.0.1:8000/controlled.html?check=1');
  assert.ok(['suspicious','below_threshold'].includes(result.state));
  assert.ok(Number.isFinite(result.risk_score)&&Number.isFinite(result.threshold));
});

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

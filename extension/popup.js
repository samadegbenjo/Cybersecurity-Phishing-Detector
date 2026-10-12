const result = document.getElementById('result');
function show(data) {
  result.replaceChildren();
  result.dataset.state=data.state;
  const lines=[data.text];
  if (data.evaluated_url) lines.push('URL evaluated: ' + data.evaluated_url);
  if (data.limitation) lines.push(data.limitation);
  if (data.observations) lines.push('URL observations: ' + JSON.stringify(data.observations));
  lines.push('Risk score: ' + (data.risk_score === undefined ? 'unavailable' : data.risk_score.toFixed(4)));
  lines.push('Model threshold: ' + (data.threshold == null ? 'unavailable' : data.threshold.toFixed(4)));
  for (const line of lines) {
    const paragraph=document.createElement('p');
    paragraph.textContent=line;
    result.appendChild(paragraph);
  }
}
async function busy(button, operation) {
  button.disabled=true;
  show({state:'pending',text:'Evaluating…'});
  try { show(await operation()); }
  catch (error) { show({state:'unable',text:'Unable to evaluate: ' + error.message}); }
  finally { button.disabled=false; }
}
document.getElementById('evaluate').addEventListener('click', event => busy(event.target, async () => {
  const [tab]=await chrome.tabs.query({active:true,currentWindow:true});
  return PhishingClient.evaluatePage(tab?.url);
}));
document.getElementById('record').addEventListener('change', event => busy(event.target, async () => {
  const file=event.target.files[0];
  if (!file || file.size>64000) throw new Error('Choose a feature-record JSON file under 64 KB');
  return PhishingClient.evaluateRecord(JSON.parse(await file.text()));
}));

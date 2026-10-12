/* Shared by popup and Node tests. Never maps an error to a benign verdict. */
(function (root) {
  const BASE = 'http://127.0.0.1:5000';
  async function post(path, body, fetcher = fetch, timeoutMs = 4000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetcher(BASE + path, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body), signal: controller.signal
      });
      if (!response.ok) throw new Error('Backend HTTP ' + response.status);
      return await response.json();
    } finally { clearTimeout(timer); }
  }
  function validPrediction(data, mode = "offline") {
    return data && data.mode === mode && data.schema_version === (mode === 'offline' ? 'offline-v1' : 'live-url-v1') &&
      typeof data.phishing === 'boolean' && typeof data.risk_score === 'number' &&
      Number.isFinite(data.risk_score) && data.risk_score >= 0 && data.risk_score <= 1 &&
      typeof data.threshold === 'number' && Number.isFinite(data.threshold) && data.threshold >= 0 && data.threshold <= 1 &&
      data.phishing === (data.risk_score >= data.threshold) &&
      data.state === (data.phishing ? 'suspicious' : 'below_threshold');
  }
  function decision(data, mode = "offline") {
    if (!validPrediction(data, mode)) throw new Error('Invalid prediction response');
    return {state: data.phishing ? 'suspicious' : 'below_threshold',
      text: (mode === 'offline' ? 'Offline record: ' : 'Active URL: ') + (data.phishing ? 'suspicious' : 'below threshold (safety unverified)'),
      risk_score: data.risk_score, threshold: data.threshold};
  }
  async function evaluateRecord(record, fetcher, timeoutMs) {
    try { return decision(await post('/predict', record, fetcher, timeoutMs)); }
    catch (error) { return {state: 'unable', text: 'Unable to evaluate: ' + error.message}; }
  }
  async function evaluatePage(url, fetcher, timeoutMs) {
    try {
      const parsed = new URL(url);
      if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('Unsupported page');
      const data = await post('/predict-url', {schema_version: 'live-url-v1', url}, fetcher, timeoutMs);
      const result = decision(data, 'live_url');
      if (data.feature_count !== 39 || typeof data.limitation !== 'string') throw new Error('Invalid live metadata');
      return {...result, evaluated_url:url, limitation:data.limitation};
    } catch (error) { return {state:'unable',text:'Unable to evaluate: ' + error.message}; }
  }
  async function observePage(url, fetcher, timeoutMs) {
    try {
      const parsed = new URL(url);
      if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('Unsupported page');
      const data = await post('/observe', {schema_version: 'url-observation-v1', url}, fetcher, timeoutMs);
      const o = data && data.observations;
      if (!data || data.state !== 'unable' || data.mode !== 'browser_demo' || data.schema_version !== 'url-observation-v1' ||
          data.risk_score !== null || !o || !Number.isInteger(o.url_length) || !Number.isInteger(o.dot_count) ||
          !Number.isInteger(o.at_count) || typeof o.uses_https !== 'boolean' || typeof data.reason !== 'string' ||
          !(data.threshold === null || (Number.isFinite(data.threshold) && data.threshold >= 0 && data.threshold <= 1))) {
        throw new Error('Invalid observation response');
      }
      return {state:'unable',text:'Unable to evaluate: ' + data.reason,observations:o,threshold:data.threshold};
    } catch (error) { return {state:'unable',text:'Unable to evaluate: ' + error.message}; }
  }
  const api={post,decision,evaluateRecord,evaluatePage,observePage};
  if (typeof module !== 'undefined' && module.exports) module.exports=api;
  else root.PhishingClient=api;
})(globalThis);

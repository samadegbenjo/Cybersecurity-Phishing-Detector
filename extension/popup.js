document.getElementById('evaluate').addEventListener('click', () => {
  // Show loading state
  const resultDiv = document.getElementById('result');
  resultDiv.innerHTML = `<p>Analyzing website...</p>`;
  
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.tabs.sendMessage(tabs[0].id, { action: 'evaluate' }, (response) => {
      if (response) {
        const resultClass = response.phishing ? 'danger' : 'safe';
        const resultText = response.phishing ? 'Phishing detected!' : 'No phishing detected.';
        
        resultDiv.innerHTML = `
          <p class="detection-result ${resultClass}">Random Forest: ${resultText}</p>
          ${response.metadata ? `
            <p class="small">URL: ${response.metadata.url}</p>
            <p class="small">Timestamp: ${response.metadata.timestamp}</p>
            <p class="small">Model: ${response.metadata.model.random_forest.name} v${response.metadata.model.random_forest.version}</p>
          ` : ''}
        `;
      } else {
        resultDiv.innerHTML = '<p class="danger">Error evaluating the page.</p>';
      }
    });
  });
});
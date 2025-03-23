chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
  if (request.action === 'evaluate') {
    const features = extractFeatures();
    const url = window.location.href;
    
    fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ 
        features,
        url 
      })
    })
    .then(response => response.json())
    .then(data => sendResponse({
      phishing: data.phishing,
      metadata: data.metadata
    }))
    .catch(error => {
      console.error('Error:', error);
      sendResponse({ phishing: false });
    });
    return true; // Keep the message channel open for sendResponse
  }
});

function extractFeatures() {
  // Implement feature extraction logic here
  // Return an array of features corresponding to the model input
  return [
    3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 25, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 17, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0.207316, 0, 60781, -1, -1, 1, 2, 0, 892, 0, 0, 0, 0, 0, 1
  ];
}
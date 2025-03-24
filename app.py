from flask import Flask, request, jsonify
import onnxruntime as rt
import numpy as np
import joblib
import datetime

app = Flask(__name__)

# Load models
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
rf_sess = rt.InferenceSession("rf_model.onnx")

# Load custom threshold
try:
    with open("threshold.txt", "r") as f:
        THRESHOLD = float(f.read().strip())
    print(f"Custom threshold loaded: {THRESHOLD}")
except FileNotFoundError:
    THRESHOLD = 0.5
    print(f"No threshold file found. Using default: {THRESHOLD}")

# Define model metadata
MODEL_METADATA = {
    'rf': {
        'name': 'Random Forest',
        'version': '1.0',
        'trained_on': '2025-03-23',
        'description': 'Ensemble of decision trees',
        'threshold': THRESHOLD
    }
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    url = request.json.get('url', 'Unknown URL')
    data = np.array(data).reshape(1, -1)
    scaled_data = scaler.transform(data)
    reduced_data = pca.transform(scaled_data)
    
    input_name = rf_sess.get_inputs()[0].name
    
    # Get probability outputs
    pred_proba = None
    output_names = [output.name for output in rf_sess.get_outputs()]
    
    # Try to find probability output (often the second output if available)
    if len(output_names) > 1:
        # First output is usually label, second is probability
        proba_output_name = output_names[1] 
        pred_proba = rf_sess.run([proba_output_name], {input_name: reduced_data.astype(np.float32)})[0][0]
        phish_prob = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
    else:
        # If only one output, assume it's the label and set probability to 1.0 or 0.0
        label_output_name = output_names[0]
        rf_pred = rf_sess.run([label_output_name], {input_name: reduced_data.astype(np.float32)})[0][0]
        phish_prob = 1.0 if rf_pred > 0 else 0.0
    
    # Apply threshold to determine final prediction
    is_phishing = int(phish_prob >= THRESHOLD)
    
    # Prepare response with metadata
    timestamp = datetime.datetime.now().isoformat()
    response = {
        'phishing': is_phishing,
        'metadata': {
            'timestamp': timestamp,
            'url': url,
            'model': {
                'random_forest': {
                    'prediction': is_phishing,
                    'raw_score': float(phish_prob),
                    'threshold_used': THRESHOLD,
                    **MODEL_METADATA['rf']
                }
            }
        }
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
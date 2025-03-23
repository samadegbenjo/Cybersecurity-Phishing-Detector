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

# Define model metadata
MODEL_METADATA = {
    'rf': {
        'name': 'Random Forest',
        'version': '1.0',
        'trained_on': '2025-03-23',
        'description': 'Ensemble of decision trees'
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
    rf_label_name = rf_sess.get_outputs()[0].name
    
    rf_pred = rf_sess.run([rf_label_name], {input_name: reduced_data.astype(np.float32)})[0]
    
    # Prepare response with metadata
    timestamp = datetime.datetime.now().isoformat()
    response = {
        'phishing': int(rf_pred[0]),
        'metadata': {
            'timestamp': timestamp,
            'url': url,
            'model': {
                'random_forest': {
                    'prediction': int(rf_pred[0]),
                    'raw_score': float(rf_pred[0]),
                    **MODEL_METADATA['rf']
                }
            }
        }
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
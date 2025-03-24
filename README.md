# Digitaltwin Phishing Detection

A browser extension that uses machine learning to detect phishing websites in real-time.

## Project Overview

Digitaltwin Phishing Detection leverages a Random Forest classifier to analyze websites and identify potential phishing attempts. The extension extracts features from the current webpage, sends them to a local server for analysis, and provides immediate feedback to the user.

## Architecture

The system consists of three main components:

1. **Chrome Extension**: Frontend interface that captures the current website's features and displays the phishing detection results.
2. **Flask Backend**: API server that processes feature data and makes predictions using the trained model.
3. **Machine Learning Model**: Random Forest classifier trained to identify phishing websites.


### Workflow:
1. User clicks "Evaluate Website" on the extension popup
2. Extension extracts features from the current webpage
3. Features are sent to the Flask server
4. Server preprocesses the data (scaling and PCA)
5. Model makes a prediction
6. Result is returned to the extension
7. User is shown the phishing detection result

## Installation

### Prerequisites
- Python 3.6+
- Chrome browser
- Pip package manager

### Step 1: Install Python Dependencies

```bash
pip install flask scikit-learn pandas numpy joblib onnx onnxruntime skl2onnx
```

### Step 2: Train the Model

1. Place your dataset CSV file in the project root directory as `dataset.csv`
2. Run the preprocessing script:
   ```bash
   python dataset_preprocessing.py
   ```
3. Run the model training script:
   ```bash
   python model_training.py
   ```

This will generate the necessary model files: `rf_model.pkl`, `rf_model.onnx`, `scaler.pkl`, `pca.pkl`, and `threshold.txt`.

### Step 3: Start the Flask Server

```bash
python app.py
```

This will start the Flask server on http://localhost:5000.

### Step 4: Load the Extension into Chrome

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" using the toggle in the top-right corner
3. Click "Load unpacked" button
4. Select the `extension` folder from the project directory
5. The extension should now be installed and visible in your browser toolbar

## Usage

1. Navigate to any website you want to check
2. Click on the Phishing Detection extension icon in your toolbar
3. Click the "Evaluate Website" button in the popup
4. Wait for the analysis to complete
5. Review the result: "Phishing detected!" or "No phishing detected"
6. Additional metadata about the analysis will be displayed below the result

## Technical Details

### Machine Learning Model
- **Algorithm**: Random Forest Classifier
- **Feature Preprocessing**: StandardScaler and PCA (Principal Component Analysis)
- **Model Format**: ONNX (Open Neural Network Exchange)
- **Threshold**: Optimized decision threshold for balanced precision/recall

### Features Extracted
The system extracts 100+ features from the website, including:
- URL characteristics
- HTML and JavaScript properties
- External link analysis
- Domain information


# Brain Tumor Detection

AI-powered brain tumor classification using fused MRI and CT scan images with 95.9% accuracy.

## Model Performance
- **Accuracy**: 95.9%
- **Architecture**: Custom CNN with image fusion
- **Input**: 224x224 RGB images
- **Classes**: Healthy, Tumor

## Quick Start

### Flask App
```bash
pip install -r flask_requirements.txt
python flask_app.py
```
Open http://127.0.0.1:5000

### Streamlit App
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `braintumor.ipynb` - Model training notebook
- `fused_cnn_model.keras` - Trained model
- `flask_app.py` - Flask web application
- `app.py` - Streamlit application

## Usage
Upload a brain scan image (MRI or CT) to get instant tumor detection results with confidence scores.

## Dataset
- MRI and CT scan images
- Binary classification: Healthy vs Tumor
- Image fusion technique for improved accuracy
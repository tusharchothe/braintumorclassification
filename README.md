
# Brain Tumor Classification

This project uses deep learning to classify brain tumor images from CT and MRI scans. It includes:
- Data preprocessing and augmentation
- CNN model training and evaluation
- Flask web app for image upload and prediction

## Project Structure
- `app.py`: Main application logic (Streamlit)
- `flask_app.py`: Flask web server for predictions
- `braintumor.ipynb`: Jupyter notebook for model development
- `Dataset/`: Contains CT and MRI images (Healthy and Tumor)
- `templates/index.html`: Web app frontend
- `requirements.txt`, `flask_requirements.txt`: Python dependencies
- `best_cnn.keras`, `fused_cnn_model.keras`: Saved models

## Usage
1. Install dependencies: `pip install -r requirements.txt` or `pip install -r flask_requirements.txt`
2. Run the Flask app: `python flask_app.py` or Streamlit app: `streamlit run app.py`
3. Access the web interface to upload images and get predictions

## Dataset
- CT and MRI images categorized as Healthy or Tumor

## Author
- GitHub: [tusharchothe](https://github.com/tusharchothe)

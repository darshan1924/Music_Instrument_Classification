# 🎸 Musical Instrument Classifier

**CNN-based image classification web app** that identifies 30 musical instruments and categorizes them as Classical, Indo-Western, or Western.

## Features
- Trained on **30 Musical Instruments dataset** (~5K images) [Kaggle]
- **MobileNetV2 transfer learning** (85-95% accuracy)
- **Responsive Streamlit web app** (mobile + desktop)
- Predicts instrument name + category

## Files
- `final_instrument_model.h5` - Trained CNN model
- `classes.npy` - 30 class labels
- `app.py` - Streamlit web application
- Colab notebook - Full training pipeline

## Demo
Upload instrument image → Get prediction with confidence + category
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory  # Fallback safe

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('final_instrument_model.h5', compile=False)
    classes = np.load('classes.npy', allow_pickle=True).tolist()
    return model, classes

model, classes = load_model()

category_map = {
    'acordian': 'Western', 'alphorn': 'Western', 'bagpipes': 'Western', 'banjo': 'Western',
    'bongo drum': 'Indo-Western', 'casaba': 'Indo-Western', 'castanets': 'Indo-Western',
    'clarinet': 'Classical', 'clavichord': 'Classical', 'concertina': 'Western',
    'Didgeridoo': 'Indo-Western', 'drums': 'Western', 'dulcimer': 'Western',
    'flute': 'Classical', 'guiro': 'Indo-Western', 'guitar': 'Western',
    'harmonica': 'Western', 'harp': 'Classical', 'marakas': 'Indo-Western',
    'ocarina': 'Western', 'piano': 'Classical', 'saxaphone': 'Western',
    'sitar': 'Indo-Western', 'steel drum': 'Indo-Western', 'Tambourine': 'Western',
    'trombone': 'Classical', 'trumpet': 'Classical', 'tuba': 'Classical',
    'violin': 'Classical', 'Xylophone': 'Western'
}

st.title("🎸 Musical Instrument Classifier")
uploaded_file = st.file_uploader("Choose image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded", use_container_width=True)
    
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array, verbose=0)
    idx = np.argmax(pred[0])
    instr = classes[idx]
    cat = category_map.get(instr.lower(), 'Unknown')
    conf = pred[0][idx] * 100
    
    st.success(f"**Instrument:** {instr}")
    st.success(f"**Confidence:** {conf:.1f}%")
    st.success(f"**Category:** {cat}")

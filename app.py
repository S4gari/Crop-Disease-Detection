import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# --------------------------- CONFIG -------------------------------------------------
IMG_SIZE = 128
MODEL_PATH = 'crop_disease_model.h5'

SEVERITY_THRESHOLDS = {
    'Severe': 90,
    'Moderate': 75,
    'Mild': 0
}

RECOMMENDATIONS = {
    'Pepper__bell___Bacterial_spot': {
        'Mild': 'Remove infected leaves',
        'Moderate': 'Apply copper fungicide weekly',
        'Severe': 'Destroy infected plants and rotate crops'
    },
    'Pepper__bell___healthy': {
        'Mild': 'No treatment needed',
        'Moderate': 'Ensure good airflow',
        'Severe': 'Regular inspection recommended'
    },
    'Potato___Early_blight': {
        'Mild': 'Use Neem oil spray twice a week',
        'Moderate': 'Use Copper-based fungicide',
        'Severe': 'Apply Mancozeb and consult an expert'
    },
    'Potato___Late_blight': {
        'Mild': 'Improve field drainage and airflow',
        'Moderate': 'Use Chlorothalonil sprays',
        'Severe': 'Apply Cymoxanil + Mancozeb and consult an expert'
    },
    'Potato___healthy': {
        'Mild': 'No action needed',
        'Moderate': 'Check regularly for early symptoms',
        'Severe': 'Maintain proper irrigation and spacing'
    },
    'Tomato___Target_Spot': {
        'Mild': 'Use baking soda solution',
        'Moderate': 'Apply fungicide like chlorothalonil',
        'Severe': 'Remove infected plants and consult an expert'
    },
    'Tomato___Tomato_mosaic_virus': {
        'Mild': 'Use virus-free seeds',
        'Moderate': 'Remove infected plants',
        'Severe': 'Disinfect tools and rotate crops'
    },
    'Tomato___Tomato_YellowLeaf__Curl_virus': {
        'Mild': 'Control whiteflies using traps',
        'Moderate': 'Apply insecticidal soap',
        'Severe': 'Remove affected plants and avoid replanting immediately'
    },
    'Tomato_Bacterial_spot': {
        'Mild': 'Remove affected leaves',
        'Moderate': 'Use copper fungicide',
        'Severe': 'Destroy affected plants, rotate crops'
    },
    'Tomato_Early_blight': {
        'Mild': 'Spray neem oil weekly',
        'Moderate': 'Use fungicide with chlorothalonil',
        'Severe': 'Apply Mancozeb and consult expert'
    },
    'Tomato_healthy': {
        'Mild': 'No action needed',
        'Moderate': 'Check leaves weekly',
        'Severe': 'Ensure proper soil and sunlight'
    },
    'Tomato_Late_blight': {
        'Mild': 'Remove wet leaves, improve airflow',
        'Moderate': 'Use protectant fungicide',
        'Severe': 'Apply metalaxyl-based fungicide'
    },
    'Tomato_Leaf_Mold': {
        'Mild': 'Use sulfur sprays',
        'Moderate': 'Apply chlorothalonil',
        'Severe': 'Increase ventilation, use resistant varieties'
    },
    'Tomato_Septoria_leaf_spot': {
        'Mild': 'Prune lower leaves',
        'Moderate': 'Apply fungicides weekly',
        'Severe': 'Destroy debris and rotate crops'
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'Mild': 'Spray water on leaves',
        'Moderate': 'Use neem or insecticidal soap',
        'Severe': 'Apply miticide and consult expert'
    },
}

FERTILIZERS = {
    'Pepper__bell': '10-10-10 NPK at 150 kg/acre every 30 days\nBuy from: www.agrostore.com/fertilizers',
    'Potato': 'Urea: 120kg/acre, DAP: 100kg/acre, MOP: 80kg/acre\nBuy from: www.agrostore.com/fertilizers',
    'Tomato': 'Compost: 5 tons/acre, NPK 19:19:19 every 2 weeks\nBuy from: www.agrostore.com/fertilizers',
}

PESTICIDES = {
    'Pepper__bell': 'Copper oxychloride, Streptomycin\nBuy from: www.agrostore.com/pesticides',
    'Potato': 'Chlorothalonil, Mancozeb\nBuy from: www.agrostore.com/pesticides',
    'Tomato': 'Copper oxychloride, Neem-based sprays, Mancozeb\nBuy from: www.agrostore.com/pesticides',
}

ORGANIC_ALTERNATIVES = {
    'Pepper__bell': 'Neem oil, Garlic-chili spray',
    'Potato': 'Neem oil, Garlic-chili spray',
    'Tomato': 'Cow urine extract, Neem soap'
}

@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

class_names = list(RECOMMENDATIONS.keys())

def get_prediction(_image):
    model = load_model_cached()
    img = _image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    probs = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs) * 100)
    pred_class = class_names[pred_idx]
    return pred_class, confidence

def determine_severity(confidence):
    for sev, thresh in SEVERITY_THRESHOLDS.items():
        if confidence >= thresh:
            return sev
    return 'Mild'

# --------------------------- STREAMLIT UI -------------------------------------------------
st.set_page_config(page_title="Crop Disease Detector", layout="centered")
st.title("🌾 Crop Disease Detection with Severity & Recommendations")

uploaded_file = st.file_uploader("📷 Upload a crop image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict & Recommend"):
        pred_class, confidence = get_prediction(image)
        severity = determine_severity(confidence)
        advice = RECOMMENDATIONS.get(pred_class, {}).get(severity, "Consult an agronomist.")

        crop_key = pred_class.split('___')[0] if '___' in pred_class else pred_class.split('_')[0]

        fert = FERTILIZERS.get(crop_key, "No fertilizer info available")
        pest = PESTICIDES.get(crop_key, "No pesticide info available")
        org_alt = ORGANIC_ALTERNATIVES.get(crop_key, "No organic alternatives available")

        st.success(f"✅ Disease: {pred_class}")
        st.info(f"🔍 Confidence: {confidence:.2f}%")
        st.warning(f"⚠️ Severity Level: {severity}")
        st.markdown(f"📌 **Recommendation:** {advice}")
        st.markdown(f"🌱 **Fertilizer:** {fert}")
        st.markdown(f"🧪 **Pesticide:** {pest}")
        st.markdown(f"🌿 **Organic Alternatives:** {org_alt}")

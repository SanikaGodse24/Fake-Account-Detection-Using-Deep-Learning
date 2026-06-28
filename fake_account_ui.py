import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved files
model = load_model("fake_account_detection_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# UI Layout
st.set_page_config(page_title="Fake Account Detection", page_icon="🔍")
st.title("🔍 Fake Social Media Account Detection")
st.write("Enter account details to check whether it is Fake or Real")

# User Inputs
followers = st.number_input("Followers", min_value=0, value=100)
following = st.number_input("Following", min_value=0, value=1000)
posts = st.number_input("Number of Posts", min_value=0, value=5)
has_profile_pic = st.selectbox("Has Profile Picture?", ["Yes", "No"])
account_age_days = st.number_input("Account Age (in days)", min_value=1, value=30)
bio = st.text_area("Profile Bio", "Follow me for free followers")

has_profile_pic = 1 if has_profile_pic == "Yes" else 0

# Predict Button
if st.button("Check Account"):
    # Text preprocessing
    bio_seq = tokenizer.texts_to_sequences([bio])
    bio_pad = pad_sequences(bio_seq, maxlen=20)

    # Numeric preprocessing
    numeric = np.array([[followers, following, posts, has_profile_pic, account_age_days]])
    numeric = scaler.transform(numeric)

    # Combine features
    input_data = np.hstack((numeric, bio_pad))

    # Prediction
    prediction = model.predict(input_data)[0][0]

    if prediction > 0.5:
        st.error("❌ This account is FAKE")
    else:
        st.success("✅ This account is REAL")

    st.write(f"Prediction Confidence: {prediction:.2f}")




#python -m streamlit run fake_account_ui.py
#py -3.11 -m notebook
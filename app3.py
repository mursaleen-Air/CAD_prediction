import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler

# --- Login imports (keep if you use login/logout) ---
from login import login, logout

# 🟡 Initialize session state at the start
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# 🔐 If not logged in, show the login page
if not st.session_state.logged_in:
    login()

# ✅ If logged in, show the main app and a logout button
else:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    import os
    import random


    def get_random_image_from_folder(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            return None, None
        selected_file = random.choice(files)
        full_path = os.path.join(folder_path, selected_file)
        return full_path, selected_file

    st.sidebar.markdown("### 🧠 Download a Random MRI Sample")

    # User selects folder
    image_type = st.sidebar.selectbox("Choose MRI Type", ["Normal", "Sick"])

    if st.sidebar.button("Show Random Sample"):
        folders = {
            "Normal": "images/normal",
            "Sick": "images/sick"
        }
        folder_path = folders[image_type]

        img_path, img_name = get_random_image_from_folder(folder_path)
        if img_path:
            st.sidebar.write(f"Random {image_type} MRI Image:")
            img = Image.open(img_path)
            st.sidebar.image(img, use_column_width=True)
            with open(img_path, "rb") as file:
                st.sidebar.download_button(
                    label=f"Download {image_type} MRI",
                    data=file,
                    file_name=img_name,
                    mime="image/jpeg"
                )
        else:
            st.sidebar.warning(f"No images found in {image_type} folder.")
    # 🎯 Your main application goes here
    st.title("Main Application")
    # Add more Streamlit widgets or pages here
    st.sidebar.markdown("### 📘 Instructions")
    st.sidebar.info("1. Upload MRI image\n2. Fill in the form\n3. Click Predict")

    # App details
    st.sidebar.markdown("### 🧠 Model Info")
    st.sidebar.caption("CNN for image, PCA + RF + DNN for decision.")

    # Footer
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        logout()

    # --- Define CNN architecture (same as training) ---
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 16 * 16, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )
        def forward(self, x):
            x = self.features(x)                     # Shape: [batch, 64, 16, 16]
            x = torch.flatten(x, 1)                  # Flatten to [batch, 16384]
            x = self.classifier[0](x)                # Pass through Linear(16384 -> 128)
            x = self.classifier[1](x)                # ReLU
            return x

        def extract_features(self, x):
            x = self.features(x)                     # shape [batch, 64, 16, 16]
            x = torch.flatten(x, 1)                  # shape [batch, 16384]
            x = self.classifier[1](x)                # Linear(16384 -> 128)
            x = self.classifier[2](x)                # ReLU
            return x


    # --- Paths to your saved models and scalers ---
    cnn_model_path = "cnn_model.pkl"
    sensor_scaler_path = "sensor_scaler3.pkl"
    mri_scaler_path = "mri_scaler2.pkl"
    sensor_pca_path = "sensor_pca3.pkl"
    mri_pca_path = "mri_pca2.pkl"

    xgb_model_path = "xgb_model2.pkl"
    dnn_model_path = "dnn_model_state_dict2.pth"
    post_pca_scaler_path = "sensor_pca_scaler.pkl"
    with open('concatenated_scaler.pkl', 'rb') as f:
        concatenated_scaler = pickle.load(f)
    with open(post_pca_scaler_path, "rb") as f:
        post_pca_scaler = pickle.load(f)
    # --- Load CNN model ---
    cnn_model = SimpleCNN()
    with open(cnn_model_path, "rb") as f:
        state_dict = pickle.load(f)
        cnn_model.load_state_dict(state_dict)
    cnn_model.eval()

    # --- Load scalers and PCA ---
    sensor_scaler = joblib.load(sensor_scaler_path)
    mri_scaler = joblib.load(mri_scaler_path)
    sensor_pca = joblib.load(sensor_pca_path)
    mri_pca = joblib.load(mri_pca_path)

    # --- Load XGBoost model ---
    xgb_model = joblib.load(xgb_model_path)

    # --- Define and load DNN model ---
    class EnsembleDNN(nn.Module):
        def __init__(self, input_dim=54):  # fused feature size
            super(EnsembleDNN, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            )
        def forward(self, x):
            return self.net(x)

    dnn_model = EnsembleDNN()
    dnn_model.load_state_dict(torch.load(dnn_model_path))
    dnn_model.eval()

    # --- Image preprocessing transform ---
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # --- Helper functions ---
    def load_image(file):
        try:
            img = Image.open(file).convert("L")
            img = img.resize((128, 128))
            return img
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None

    def extract_cnn_features(img):
        img_tensor = transform(img).unsqueeze(0)  # [1,1,128,128]
        with torch.no_grad():
            features = cnn_model(img_tensor).cpu().numpy()  # Now shape is (1, 128)
        assert features.shape[1] == 128, "Unexpected CNN feature size"
        return features

    def preprocess_sensor_data(inputs):
        sensor_scaled = sensor_scaler.transform(inputs)
        sensor_reduced = sensor_pca.transform(sensor_scaled)
        return sensor_reduced

    def preprocess_mri_data(features):
        pca_features = mri_pca.transform(features)
        scaled = mri_scaler.transform(pca_features)
        return scaled

    # --- Streamlit user input ---
    st.title("🩺 Heart Disease Predictor")

    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = load_image(uploaded_file)
        if img:
            st.image(img, caption="Uploaded MRI Image", use_container_width=True)
    else:
        st.warning("Upload an MRI image to proceed.")

    st.subheader("🔢 Enter Patient Data")

    cols = st.columns(2)

    age = st.slider("Age", 0, 100, 50)
    sex = st.radio("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.slider("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.slider("Cholesterol", 100, 500, 250)
    fasting_bs = st.radio("Fasting Blood Sugar > 120?", ["Yes", "No"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.radio("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    # Encoding
    sex_val = 1 if sex == "Male" else 0
    fasting_bs_val = 1 if fasting_bs == "Yes" else 0
    exercise_angina_val = 1 if exercise_angina == "Yes" else 0

    # Label encoding categorical features
    chest_pain_dict = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
    resting_ecg_dict = {"Normal": 0, "ST": 1, "LVH": 2}
    st_slope_dict = {"Up": 0, "Flat": 1, "Down": 2}

    cp_val = chest_pain_dict[chest_pain]
    ecg_val = resting_ecg_dict[resting_ecg]
    slope_val = st_slope_dict[st_slope]

    # Combine sensor input features (total 11 now):
    sensor_features = [
        age,
        sex_val,
        resting_bp,
        cholesterol,
        fasting_bs_val,
        max_hr,
        exercise_angina_val,
        oldpeak,
        cp_val,
        ecg_val,
        slope_val
    ]
    sensor_input = np.array([sensor_features])
    # st.write("Sensor features (length):", len(sensor_features))
    # st.write("Sensor input shape:", sensor_input.shape)

    assert len(sensor_features) == 11, f"Expected 15 sensor features but got {len(sensor_features)}"
    if st.button("Predict"):
        if uploaded_file is None:
            st.error("Please upload an MRI image before prediction.")
        else:
            try:
                # CNN features from MRI image (shape: [1, 16384])
                img = Image.open(uploaded_file).convert("L")
                img_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    cnn_features = cnn_model.extract_features(img_tensor).cpu().numpy()

                # 1) Scale CNN features first, then PCA reduce them
                
                cnn_features_scaled = mri_scaler.transform(cnn_features)        # scale first
                features_mri_pca = mri_pca.transform(cnn_features_scaled)       # then PCA reduce

                # Check MRI PCA output shape (should be e.g. [1, 42] or as your PCA was fitted)
                #st.write("MRI PCA features shape:", features_mri_pca.shape)

                # Preprocess sensor data: scale then PCA
                sensor_input = np.array([sensor_features])
                
                sensor_scaled = sensor_scaler.transform(sensor_input)
                sensor_pca_features = sensor_pca.transform(sensor_scaled)
                #sensor_pca_features = post_pca_scaler.transform(sensor_pca_features)  #

                # Check sensor PCA output shape (e.g. [1, 15] or as fitted)
                #st.write("Sensor PCA features shape:", sensor_pca_features.shape)

                # Combine PCA-reduced sensor + MRI features
                combined = np.concatenate([sensor_pca_features, features_mri_pca], axis=1)
                #combined_scaled = concatenated_scaler.transform(combined)

                #st.write("Combined feature shape:", combined.shape)

                # Assert combined feature length matches DNN input
                assert combined.shape[1] == 54, f"Expected combined feature length 53, got {combined.shape[1]}"

                # Predict with XGBoost
                xgb_pred_prob = xgb_model.predict_proba(combined)[:, 1]

                # Predict with DNN
                combined_tensor = torch.tensor(combined, dtype=torch.float32)
                with torch.no_grad():
                    dnn_logits = dnn_model(combined_tensor)
                    raw_logit = dnn_logits.item()
                    dnn_pred_prob = torch.sigmoid(dnn_logits).cpu().numpy().flatten()
                    prob_value = dnn_pred_prob[0]

                    # st.write(f"Raw DNN logit: {raw_logit:.4f}")
                    # st.write(f"DNN predicted probability: {prob_value:.4f}")


                # Ensemble average
                ensemble_prob = (0.2 * xgb_pred_prob + 0.8 * dnn_pred_prob)
                ensemble_prob = 1 - ensemble_prob
                pred_label = "Sick" if ensemble_prob[0] > 0.6 else "Normal"
                st.success(f"Prediction: **{pred_label}**")
                # st.write("XGBoost input shape:", combined.shape)
                #st.write("XGBoost prediction probabilities:", xgb_pred_prob)
                #st.write("DNN logits:", dnn_logits)
                #st.write("DNN prediction probabilities:", dnn_pred_prob)
                #st.write("Ensembled probabilities:", ensemble_prob)
                # st.write("Combined feature vector sample:")
                # st.write(combined)
                # st.write("MRI PCA features after fix:", features_mri_pca)
                # st.write("Mean:", np.mean(features_mri_pca), "Std:", np.std(features_mri_pca))










            except Exception as e:
                st.error(f"Prediction failed: {e}")



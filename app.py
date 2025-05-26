import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import base64
from io import BytesIO
from huggingface_hub import hf_hub_download

# Định nghĩa tham số
INCEPTION_MODEL_PATH = hf_hub_download(repo_id="HakuDevon/DogBreed", filename="dog_breed_classifier.h5")
CNN_MODEL_PATH = hf_hub_download(repo_id="HakuDevon/DogBreed", filename="dog_breed_classifier_cnn.h5")
CLASS_NAMES_PATH = 'class_names.txt'
INCEPTION_HISTORY_PATH = 'inception_history.pkl'
CNN_HISTORY_PATH = 'cnn_history.pkl'
PREDICTION_HISTORY_PATH = 'prediction_history.pkl'

# Tải danh sách giống chó từ file
if not os.path.exists(CLASS_NAMES_PATH):
    st.error(f"Class names file {CLASS_NAMES_PATH} not found.")
    st.stop()
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Tải cả hai mô hình đã huấn luyện
@st.cache_resource
def load_models():
    models = {}
    if not os.path.exists(INCEPTION_MODEL_PATH):
        st.error(f"InceptionV3 model file {INCEPTION_MODEL_PATH} not found.")
        st.stop()
    models['InceptionV3'] = tf.keras.models.load_model(INCEPTION_MODEL_PATH)
    if not os.path.exists(CNN_MODEL_PATH):
        st.error(f"CNN model file {CNN_MODEL_PATH} not found. ")
        st.stop()
    models['CNN'] = tf.keras.models.load_model(CNN_MODEL_PATH)
    return models

models = load_models()

# Hàm dự đoán trên một ảnh
def predict_image(image, model, class_names):
    img_size = (299, 299) if model_choice == 'InceptionV3' else (224, 224)
    img = image.resize(img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_breed = class_names[predicted_class].split('-')[1] if '-' in class_names[predicted_class] else class_names[predicted_class]
    probabilities = prediction[0]
    return predicted_breed, probabilities, predicted_class

# Hàm chuyển ảnh thành base64 để lưu
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Hàm chuyển base64 thành ảnh
def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

# Hàm lưu lịch sử dự đoán vào file .pkl
def save_prediction_history(prediction_history):
    try:
        with open(PREDICTION_HISTORY_PATH, 'wb') as f:
            pickle.dump(prediction_history, f)
    except Exception as e:
        st.error(f"Error saving prediction history: {e}")

# Hàm tải lịch sử dự đoán từ file .pkl
def load_prediction_history():
    if os.path.exists(PREDICTION_HISTORY_PATH):
        try:
            with open(PREDICTION_HISTORY_PATH, 'rb') as f:
                history = pickle.load(f)
                if not history:
                    return []
                return history
        except Exception as e:
            st.error(f"Error reading prediction history file: {e}")
            with open(PREDICTION_HISTORY_PATH, 'wb') as f:
                pickle.dump([], f)
            return []
    with open(PREDICTION_HISTORY_PATH, 'wb') as f:
        pickle.dump([], f)
    return []

# Tải dữ liệu lịch sử huấn luyện từ file .pkl
@st.cache_data
def load_training_history(model_choice):
    history_path = INCEPTION_HISTORY_PATH if model_choice == 'InceptionV3' else CNN_HISTORY_PATH
    placeholder_data = {
        'loss': [0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23, 0.2, 0.18],
        'val_loss': [0.55, 0.5, 0.48, 0.45, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38],
        'accuracy': [0.65, 0.68, 0.72, 0.75, 0.78, 0.8, 0.82, 0.83, 0.85, 0.86],
        'val_accuracy': [0.62, 0.65, 0.67, 0.69, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76]
    }
    placeholder_fine_data = {
        'loss': [0.18, 0.16, 0.14, 0.12, 0.1],
        'val_loss': [0.38, 0.37, 0.36, 0.35, 0.34],
        'accuracy': [0.86, 0.87, 0.88, 0.89, 0.9],
        'val_accuracy': [0.76, 0.77, 0.78, 0.79, 0.8]
    }

    if os.path.exists(history_path):
        try:
            with open(history_path, 'rb') as f:
                training_history = pickle.load(f)
            history_data = training_history.get('initial', training_history if 'loss' in training_history else placeholder_data)
            history_fine_data = training_history.get('fine_tuning', placeholder_fine_data)
            return history_data, history_fine_data
        except Exception as e:
            st.warning(f"Error reading {history_path}: {e}. Using placeholder data.")
    else:
        st.warning(f"{history_path} not found. Using placeholder data.")
    return placeholder_data, placeholder_fine_data

# Tải lịch sử dự đoán vào session_state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = load_prediction_history()

# Lưu trữ trạng thái ảnh đã tải lên và kết quả dự đoán
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Theo dõi trạng thái hiển thị chi tiết trong tab History
if 'show_details' not in st.session_state:
    st.session_state.show_details = {}

# Tạo tabs cho giao diện
tab1, tab2, tab3, tab4 = st.tabs(["Classifier", "Charts", "Statistics", "History"])

# Tab 1: Classifier
with tab1:
    st.title("Dog Breed Classifier")

    # Combobox chọn mô hình
    model_choice = st.selectbox("Select Model", ["InceptionV3", "CNN"], key="model_select")
    selected_model = models[model_choice]

    # Tải nhiều ảnh cùng lúc
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="file_uploader")

    # Hiển thị các ảnh đã tải lên với chiều cao bằng nhau
    if uploaded_files:
        cols_per_row = 4
        st.markdown("""
            <style>
            .image-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            }
            .image-container img {
                width: 150px !important;
                height: 150px !important;
                object-fit: cover;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
            .image-container p {
                margin: 5px 0 0 0;
                font-size: 12px;
                word-wrap: break-word;
                max-width: 150px;
            }
            .prediction-container {
                display: flex;
                align-items: stretch;
                gap: 20px;
                margin-bottom: 20px;
            }
            .prediction-image {
                flex: 1;
                min-width: 300px;
            }
            .prediction-image img {
                width: 100%;
                height: auto;
                max-height: 400px;
                object-fit: contain;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
            .prediction-details {
                flex: 2;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            </style>
        """, unsafe_allow_html=True)

        for i in range(0, len(uploaded_files), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, file in enumerate(uploaded_files[i:i + cols_per_row]):
                with cols[j]:
                    image = Image.open(file).convert("RGB")
                    aspect_ratio = image.width / image.height
                    new_width = int(150 * aspect_ratio)
                    image = image.resize((new_width, 150), Image.LANCZOS)
                    st.image(image, caption=file.name, output_format="PNG")
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    # Nút dự đoán
    if st.button("Predict") and uploaded_files:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total_images = len(uploaded_files)
        st.session_state.predictions = []  # Xóa kết quả cũ

        for idx, uploaded_file in enumerate(uploaded_files):
            progress_text.text(f"Predicting image {idx + 1} of {total_images}...")
            progress_bar.progress((idx + 1) / total_images)

            image = Image.open(uploaded_file).convert("RGB")
            breed, probabilities, predicted_class = predict_image(image, selected_model, class_names)
            confidence = probabilities[predicted_class]

            top_k = 5
            top_indices = np.argsort(probabilities)[-top_k:][::-1]
            top_labels = [class_names[i].split('-')[1] if '-' in class_names[i] else class_names[i] for i in top_indices]
            top_probs = [probabilities[i] for i in top_indices]

            # Lưu dự đoán vào session_state
            image_resized = image.resize((100, 100))
            image_base64 = image_to_base64(image_resized)
            prediction_entry = {
                "image": image_base64,
                "breed": breed,
                "confidence": float(confidence),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "feedback": "Not rated",
                "filename": uploaded_file.name,
                "top_labels": top_labels,
                "top_probs": top_probs,
                "model": model_choice
            }
            st.session_state.predictions.append(prediction_entry)
            st.session_state.prediction_history.append(prediction_entry)

        # Lưu lịch sử dự đoán sau khi hoàn tất
        save_prediction_history(st.session_state.prediction_history)
        progress_text.empty()
        progress_bar.empty()

    # Hiển thị kết quả dự đoán và feedback
    if st.session_state.predictions:
        for idx, prediction in enumerate(st.session_state.predictions):
            st.subheader(f"Prediction for {prediction['filename']} (Model: {prediction['model']})")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(base64_to_image(prediction['image']), use_container_width=True)
            with col2:
                st.write(f"**Breed:** {prediction['breed']}")
                st.write(f"**Confidence:** {prediction['confidence']:.2%}")

                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.barh(prediction['top_labels'][::-1], prediction['top_probs'][::-1], color=['#ff9999', '#ffcc99', '#ffeb99', '#ccff99', '#99ffcc'])
                ax.set_xlabel("Probability")
                ax.set_title("Top 5 Predictions")
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2%}', va='center')
                plt.tight_layout()
                st.pyplot(fig)

            # Feedback
            feedback_key = f"feedback_{len(st.session_state.prediction_history)-len(st.session_state.predictions)+idx}"
            feedback = st.radio(
                f"Is this prediction correct for {prediction['filename']}?",
                ("Not rated", "Yes", "No"),
                key=feedback_key,
                horizontal=True
            )

            # Cập nhật feedback ngay khi thay đổi
            if feedback != st.session_state.prediction_history[len(st.session_state.prediction_history)-len(st.session_state.predictions)+idx]["feedback"]:
                st.session_state.prediction_history[len(st.session_state.prediction_history)-len(st.session_state.predictions)+idx]["feedback"] = feedback
                st.session_state.predictions[idx]["feedback"] = feedback
                save_prediction_history(st.session_state.prediction_history)
                st.success(f"Feedback for {prediction['filename']} saved: {feedback}")

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Charts
with tab2:
    st.title("Training Charts")

    # Tải lịch sử huấn luyện dựa trên mô hình đã chọn
    history_data, history_fine_data = load_training_history(st.session_state.get('model_select', 'InceptionV3'))

    st.subheader(f"{st.session_state.get('model_select', 'InceptionV3')} Initial Training")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history_data['loss'], label='Train Loss', color='blue')
    ax1.plot(history_data['val_loss'], label='Validation Loss', color='orange')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(history_data['accuracy'], label='Train Accuracy', color='blue')
    ax2.plot(history_data['val_accuracy'], label='Validation Accuracy', color='orange')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig)

    if 'loss' in history_fine_data:
        st.subheader(f"{st.session_state.get('model_select', 'InceptionV3')} Fine-Tuning")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(history_fine_data['loss'], label='Train Loss (Fine-tuning)', color='blue')
        ax1.plot(history_fine_data['val_loss'], label='Validation Loss (Fine-tuning)', color='orange')
        ax1.set_title('Model Loss (Fine-tuning)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax2.plot(history_fine_data['accuracy'], label='Train Accuracy (Fine-tuning)', color='blue')
        ax2.plot(history_fine_data['val_accuracy'], label='Validation Accuracy (Fine-tuning)', color='orange')
        ax2.set_title('Model Accuracy (Fine-tuning)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No fine-tuning data available.")

# Tab 3: Statistics
@st.cache_data
def compute_statistics(prediction_history):
    total_predictions = len(prediction_history)
    correct = sum(1 for entry in prediction_history if entry['feedback'] == "Yes")
    incorrect = sum(1 for entry in prediction_history if entry['feedback'] == "No")
    not_rated = sum(1 for entry in prediction_history if entry['feedback'] == "Not rated")
    
    breed_counts = {}
    for entry in prediction_history:
        breed = entry['breed']
        breed_counts[breed] = breed_counts.get(breed, 0) + 1
    top_breeds = sorted(breed_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return total_predictions, correct, incorrect, not_rated, top_breeds

with tab3:
    st.title("Statistics")

    if st.session_state.prediction_history:
        total_predictions, correct, incorrect, not_rated, top_breeds = compute_statistics(st.session_state.prediction_history)

        st.write(f"Total Predictions: {total_predictions}")
        st.write(f"Correct Predictions: {correct}")
        st.write(f"Incorrect Predictions: {incorrect}")
        st.write(f"Not Rated: {not_rated}")

        st.markdown("""
            <style>
            .chart-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: space-between;
            }
            .chart {
                flex: 1;
                min-width: 300px;
                max-width: 500px;
            }
            @media (max-width: 800px) {
                .chart {
                    flex: 100%;
                }
            }
            </style>
        """, unsafe_allow_html=True)

        if correct + incorrect > 0 or top_breeds:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])

            with col1:
                if top_breeds:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    breeds, counts = zip(*top_breeds)
                    ax.bar(breeds, counts, color='#74ebd5')
                    ax.set_title("Top 5 Predicted Breeds")
                    ax.set_xlabel("Breed")
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
            with col2:
                if correct + incorrect > 0:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.pie([correct, incorrect], labels=['Correct', 'Incorrect'], autopct='%1.1f%%', colors=['#99ff99', '#ff9999'])
                    ax.set_title("Prediction Accuracy")
                    st.pyplot(fig)

            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No predictions yet. Make a prediction in the Classifier tab to see statistics.")

# Tab 4: History
with tab4:
    st.title("Prediction History")

    if st.session_state.prediction_history:
        st.subheader("Prediction Log")
        for idx, entry in enumerate(reversed(st.session_state.prediction_history)):
            col1, col2 = st.columns([1, 3])
            with col1:
                image = base64_to_image(entry['image'])
                st.image(image, width=50)
            with col2:
                st.write(f"Breed: {entry['breed']}, Confidence: {entry['confidence']:.2%}, Timestamp: {entry['timestamp']}, Feedback: {entry['feedback']}, Model: {entry['model']}")
                detail_key = f"detail_{idx}"
                button_label = "Hide Details" if st.session_state.show_details.get(detail_key, False) else "View Details"
                if st.button(button_label, key=f"toggle_{idx}"):
                    st.session_state.show_details[detail_key] = not st.session_state.show_details.get(detail_key, False)
                    st.rerun()
                if st.session_state.show_details.get(detail_key, False):
                    with st.container():
                        st.subheader("Selected Prediction Details")
                        st.image(base64_to_image(entry['image']), caption="Selected Image", width=200)
                        st.write(f"Breed: {entry['breed']}")
                        st.write(f"Confidence: {entry['confidence']:.2%}")
                        st.write(f"Timestamp: {entry['timestamp']}")
                        st.write(f"Feedback: {entry['feedback']}")
                        st.write(f"Model: {entry['model']}")

        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.session_state.predictions = []
            st.session_state.show_details = {}
            save_prediction_history(st.session_state.prediction_history)
            st.rerun()
    else:
        st.info("No predictions yet. Upload an image in the Classifier tab to start!")

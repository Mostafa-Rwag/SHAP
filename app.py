from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import shap
from PIL import Image

app = Flask(__name__)

# تحميل نموذج ResNet50
model = resnet50.ResNet50(weights="imagenet")

# وظيفة لمعالجة الصور
def load_and_preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    # تحميل الصورة من الطلب
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    image = Image.open(file.stream)

    # معالجة الصورة والتنبؤ
    image_array = load_and_preprocess_image(image)
    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # تفسير باستخدام SHAP
    explainer = shap.GradientExplainer((model, model.input), image_array)
    shap_values = explainer.shap_values(image_array)

    # نتائج التنبؤ
    results = [
        {"label": label, "score": float(score)}
        for (_, label, score) in decoded_predictions
    ]

    return jsonify({
        'predictions': results,
        'shap_values': shap_values[0].tolist()  # قيم SHAP (يمكنك تحسين الإخراج)
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)

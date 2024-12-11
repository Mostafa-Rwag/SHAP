import os
import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

# إعدادات البيئة لتقليل رسائل التحذير واستخدام CPU فقط
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel('ERROR')

# إنشاء تطبيق Flask
app = Flask(__name__)

# تحميل نموذج ResNet50
model = resnet50.ResNet50(weights="imagenet")

# وظيفة لتحميل ومعالجة الصور
def load_and_preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    # التحقق من وجود ملف الصورة في الطلب
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # قراءة الصورة من الطلب
    file = request.files['image']
    try:
        image = Image.open(file.stream)
    except Exception as e:
        return jsonify({'error': f'Invalid image: {str(e)}'}), 400

    # معالجة الصورة
    image_array = load_and_preprocess_image(image)

    # التنبؤ بالفئات
    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # تنسيق النتائج للإرجاع
    results = [
        {"label": label, "score": float(score)}
        for (_, label, score) in decoded_predictions
    ]

    return jsonify({'predictions': results})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)

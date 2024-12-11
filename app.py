from flask import Flask, request, jsonify
import os
import numpy as np
from PIL import Image
import shap

app = Flask(__name__)

# إنشاء مجلد لتخزين الصور إذا لم يكن موجوداً
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# تحديد المكان الذي سيتم فيه تخزين الملفات المرفوعة
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return "Welcome to the SHAP Image Explanation API!"

# وظيفة لتفسير الصور باستخدام SHAP
def explain_with_shap(image_path):
    try:
        # قراءة الصورة
        image = Image.open(image_path).resize((32, 32))  # تقليص الصورة لسهولة التحليل
        image_array = np.array(image) / 255.0  # تحويل الصورة إلى قيم بين 0 و 1
        image_array = image_array.flatten()  # تحويل الصورة إلى مصفوفة 1D

        # نموذج افتراضي لتحليل SHAP (استبدل هذا بالنموذج الفعلي الخاص بك إذا كان موجوداً)
        def dummy_model(input_array):
            # نموذج بسيط يرجع متوسط قيم الإدخال لكل صورة
            return np.mean(input_array, axis=1, keepdims=True)

        # إعداد SHAP Explainer
        explainer = shap.KernelExplainer(dummy_model, np.zeros((1, image_array.size)))

        # تفسير الصورة
        shap_values = explainer.shap_values(np.expand_dims(image_array, axis=0))

        # إرجاع النتائج
        return {"shap_values": shap_values[0].tolist()}
    except Exception as e:
        return {"error": "Error in SHAP explanation", "details": str(e)}

# مسار رفع الصورة وتحليلها
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # تفسير الصورة باستخدام SHAP
        shap_results = explain_with_shap(file_path)

        return jsonify({
            "message": "File uploaded successfully",
            "file_path": file_path,
            "shap_explanation": shap_results
        }), 200

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)

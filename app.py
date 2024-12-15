# Install dependencies: Flask, SHAP, TensorFlow, and Matplotlib
from flask import Flask, jsonify
import shap
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import xception
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import urllib.request
import os

app = Flask(__name__)

# Load the pre-trained Xception model
model = xception.Xception(weights='imagenet')

# SHAP requires a custom masking function for image data
explainer = shap.Explainer(model, shap.maskers.Image("inpaint_telea", (299, 299, 3)))

# Define /predict endpoint
@app.route('/predict', methods=['GET'])
def predict():
    # Define the URL of the image to download
    image_url = "https://images.rawpixel.com/image_png_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIzLTA4L3Jhd3BpeGVsX29mZmljZV8zMF9hX3N0dWRpb19zaG90X29mX2NhdF93YXZpbmdfaW1hZ2VzZnVsbF9ib2R5X182YzRmM2YyOC0wMGJjLTQzNTYtYjM3ZC05NDM0NTgwY2FmNDcucG5n.png"

    # Download and preprocess the image
    file_path = "downloaded_image.png"
    urllib.request.urlretrieve(image_url, file_path)
    image_array, original_image = load_and_preprocess_image(file_path)

    # Add batch dimension and make predictions
    image_batch = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_batch)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Explain the model's predictions with SHAP
    shap_values = explainer(image_batch)

    # Save SHAP explanation as a visualization
    shap_image_path = os.path.join('static', 'shap_output.png')
    os.makedirs('static', exist_ok=True)
    shap.image_plot(shap_values, np.array([image_array]), show=False)
    plt.savefig(shap_image_path)
    plt.close()

    # Return predictions and SHAP visualization URL
    return jsonify({
        "predictions": decoded_predictions,
        "shap_image_url": shap_image_path
    })

def load_and_preprocess_image(image_path):
    """
    Loads and preprocesses an image for the Xception model.
    """
    image = load_img(image_path, target_size=(299, 299))
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)  # Preprocess image as required by Xception
    return image_array, image

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

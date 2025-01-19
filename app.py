from flask import Flask, render_template, request
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import joblib

app = Flask(__name__)

# Load the Random Forest model
rf_classifier = joblib.load('random_forest_vitamin_bnew.pkl')

# Load the pre-trained VGG16 model for feature extraction
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.output)

# Freeze the VGG16 layers
for layer in feature_extractor.layers:
    layer.trainable = False

# Define class names
class_names = {
    0: 'Vitamin A Deficiency',
    1: 'Vitamin B Deficiency',
    2: 'Vitamin C Deficiency',
    3: 'Vitamin D Deficiency',
    4: 'Vitamin E Deficiency',
    5: 'No Deficiency'
}

def predict_deficiency(img):
    """Predict the vitamin deficiency from an image."""
    img = img.resize((150, 150))  # Resize to match model input
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features using VGG16
    features = feature_extractor.predict(img_array)
    features_flattened = features.reshape(features.shape[0], -1)

    # Make prediction using Random Forest classifier
    prediction = rf_classifier.predict(features_flattened)
    return class_names.get(prediction[0], 'Unknown Deficiency')

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main page and handle image upload for prediction."""
    prediction = None
    if request.method == 'POST':
        file = request.files.get('image')  # Use .get() to avoid errors if the key doesn't exist
        if file:
            try:
                # Open the image file
                img = Image.open(file.stream)

                # Predict deficiency
                prediction = predict_deficiency(img)
            except Exception as e:
                print(f"Error processing image: {e}")
                prediction = "Error during prediction"
    return render_template('index.html', prediction=prediction)
if __name__=="__main__":
    app.run()

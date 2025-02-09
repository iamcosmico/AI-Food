from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['TEMPLATES_AUTO_RELOAD'] = True
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the model path
model_path = 'trained_plant_disease_model.keras'

# Set the environment variable to use CPU if GPU is not available
if not tf.config.list_physical_devices('GPU'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the trained model
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Class names for the predictions
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# Disease information dictionary (abbreviated for brevity)
disease_info = {
    'Apple___Apple_scab': {
        'harmfulness': 6,
        'recommendations': [
            "Remove and destroy infected leaves.",
            "Prune trees to improve air circulation.",
            "Apply appropriate fungicides.",
            "Monitor trees regularly for signs of disease."
        ]
    },
    'Apple___Black_rot': {
        'harmfulness': 7,
        'recommendations': [
            "Remove and destroy infected fruit, leaves, and branches.",
            "Use fungicides and practice crop rotation.",
            "Ensure proper sanitation in the orchard.",
            "Avoid wounding the trees."
        ]
    },
    'Apple___Cedar_apple_rust': {
        'harmfulness': 5,
        'recommendations': [
            "Remove galls from nearby cedar trees.",
            "Apply fungicides.",
            "Plant resistant apple varieties.",
            "Maintain proper tree spacing."
        ]
    },
    'Apple___healthy': {
        'harmfulness': 0,
        'recommendations': [
            "No action needed. Your apple tree is healthy!"
        ]
    },
    'Blueberry___healthy': {
        'harmfulness': 0,
        'recommendations': [
            "No action needed. Your blueberry plant is healthy!"
        ]
    },
    'Cherry_(including_sour)___healthy': {
        'harmfulness': 0,
        'recommendations': [
            "No action needed. Your cherry plant is healthy!"
        ]
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'harmfulness': 6,
        'recommendations': [
            "Apply sulfur or potassium bicarbonate-based fungicides.",
            "Ensure good air circulation around the plant.",
            "Prune infected leaves and shoots.",
            "Avoid excessive nitrogen fertilization."
        ]
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'harmfulness': 8,
        'recommendations': [
            "Rotate crops to reduce disease pressure.",
            "Use resistant hybrids or varieties.",
            "Apply fungicides during the early stages of disease.",
            "Ensure proper crop residue management."
        ]
    },
    'Corn_(maize)___Common_rust_': {
        'harmfulness': 6,
        'recommendations': [
            "Use resistant hybrids.",
            "Apply fungicides if the disease is severe.",
            "Remove and destroy heavily infected plants.",
            "Maintain proper crop rotation and field hygiene."
        ]
    },
    'Corn_(maize)___healthy': {
        'harmfulness': 0,
        'recommendations': [
            "No action needed. Your corn plant is healthy!"
        ]
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'harmfulness': 7,
        'recommendations': [
            "Plant resistant hybrids.",
            "Rotate crops to minimize pathogen survival.",
            "Apply fungicides if necessary.",
            "Avoid dense planting to improve air circulation."
        ]
    },
    'Grape___Black_rot': {
        'harmfulness': 8,
        'recommendations': [
            "Remove and destroy infected leaves and fruits.",
            "Apply fungicides regularly during the growing season.",
            "Ensure proper pruning for good air circulation.",
            "Avoid overhead irrigation to reduce leaf wetness."
        ]
    },
    'Grape___Esca_(Black_Measles)': {
        'harmfulness': 7,
        'recommendations': [
            "Prune and destroy infected wood.",
            "Apply fungicides to protect young vines.",
            "Avoid wounding during pruning or cultivation.",
            "Maintain healthy vine growth to reduce stress."
        ]
    },
    'Grape___healthy': {
        'harmfulness': 0,
        'recommendations': [
            "No action needed. Your grapevine is healthy!"
        ]
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'harmfulness': 10,
        'recommendations': [
            "Remove and destroy infected trees immediately.",
            "Control psyllid populations using insecticides.",
            "Plant disease-free nursery stock.",
            "Use resistant rootstocks if available."
        ]
    },
    'Peach___Bacterial_spot': {
        'harmfulness': 7,
        'recommendations': [
            "Apply copper-based sprays during the growing season.",
            "Remove and destroy infected leaves and fruits.",
            "Choose resistant varieties where possible.",
            "Avoid overhead irrigation to minimize leaf wetness."
        ]
    },
    'Peach___healthy': {
        'harmfulness': 0,
        'recommendations': [
            "No action needed. Your peach plant is healthy!"
        ]
    },
    'Tomato___Bacterial_spot': {
        'harmfulness': 6,
        'recommendations': [
            "Apply copper-based bactericides.",
            "Remove infected leaves and avoid working with wet plants.",
            "Use certified disease-free seeds and transplants.",
            "Practice crop rotation with non-susceptible crops."
        ]
    },
    'Tomato___Late_blight': {
        'harmfulness': 9,
        'recommendations': [
            "Apply fungicides immediately upon detection.",
            "Remove and destroy infected plants.",
            "Avoid planting tomatoes near potatoes, as both are hosts.",
            "Ensure proper spacing to promote air circulation."
        ]
    },
    'Tomato___healthy': {
        'harmfulness': 0,
        'recommendations': [
            "No action needed. Your tomato plant is healthy!"
        ]
    },
        'Pepper,_bell___Bacterial_spot': {
        'harmfulness': 6,
        'recommendations': [
            "Apply copper-based sprays.",
            "Remove and destroy infected plants.",
            "Use resistant varieties where possible.",
            "Practice crop rotation to reduce disease persistence."
        ]
    },
    'Pepper,_bell___healthy': {
        'harmfulness': 0,
        'recommendations': [
            "No action needed. Your bell pepper plant is healthy!"
        ]
    },
    'Potato___Early_blight': {
        'harmfulness': 7,
        'recommendations': [
            "Apply fungicides early in the season.",
            "Remove infected leaves and debris.",
            "Use certified disease-free seeds.",
            "Practice crop rotation to reduce pathogen buildup."
        ]
    },
    'Potato___healthy': {
        'harmfulness': 0,
        'recommendations': [
            "No action needed. Your potato plant is healthy!"
        ]
    },
    'Potato___Late_blight': {
        'harmfulness': 9,
        'recommendations': [
            "Apply fungicides immediately upon detection.",
            "Remove and destroy infected plants.",
            "Avoid overhead irrigation to reduce leaf wetness.",
            "Plant resistant potato varieties if available."
        ]
    },
    'Raspberry___healthy': {
        'harmfulness': 0,
        'recommendations': [
            "No action needed. Your raspberry plant is healthy!"
        ]
    },
    'Soybean___healthy': {
        'harmfulness': 0,
        'recommendations': [
            "No action needed. Your soybean plant is healthy!"
        ]
    },
    'Squash___Powdery_mildew': {
        'harmfulness': 6,
        'recommendations': [
            "Apply sulfur-based fungicides.",
            "Ensure good air circulation around the plants.",
            "Water plants at the base to avoid wetting leaves.",
            "Remove and destroy infected leaves."
        ]
    },
    'Strawberry___healthy': {
        'harmfulness': 0,
        'recommendations': [
            "No action needed. Your strawberry plant is healthy!"
        ]
    },
    'Strawberry___Leaf_scorch': {
        'harmfulness': 7,
        'recommendations': [
            "Remove and destroy infected leaves.",
            "Avoid overhead irrigation to keep leaves dry.",
            "Apply fungicides if the infection is severe.",
            "Ensure proper plant spacing for air circulation."
        ]
    },
    'Tomato___Early_blight': {
        'harmfulness': 8,
        'recommendations': [
            "Apply fungicides early in the growing season.",
            "Remove infected leaves and debris.",
            "Practice crop rotation to reduce disease persistence.",
            "Use resistant tomato varieties if available."
        ]
    },
    'Tomato___Leaf_Mold': {
        'harmfulness': 7,
        'recommendations': [
            "Remove and destroy infected leaves.",
            "Apply fungicides to control the disease.",
            "Ensure proper ventilation in greenhouses.",
            "Avoid overhead irrigation to reduce humidity."
        ]
    },
    'Tomato___Septoria_leaf_spot': {
        'harmfulness': 7,
        'recommendations': [
            "Apply fungicides as soon as symptoms appear.",
            "Remove infected leaves and debris.",
            "Practice crop rotation with non-susceptible crops.",
            "Ensure proper spacing to promote air circulation."
        ]
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'harmfulness': 6,
        'recommendations': [
            "Use miticides to control infestations.",
            "Introduce natural predators like ladybugs.",
            "Avoid water stress in plants.",
            "Remove heavily infested leaves."
        ]
    },
    'Tomato___Target_Spot': {
        'harmfulness': 7,
        'recommendations': [
            "Apply fungicides early in the infection.",
            "Remove infected leaves and debris.",
            "Ensure proper plant spacing for air circulation.",
            "Use resistant varieties if available."
        ]
    },
    'Tomato___Tomato_mosaic_virus': {
        'harmfulness': 9,
        'recommendations': [
            "Remove and destroy infected plants.",
            "Use certified virus-free seeds.",
            "Disinfect tools and equipment regularly.",
            "Avoid handling plants when wet."
        ]
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'harmfulness': 9,
        'recommendations': [
            "Control whiteflies using insecticides.",
            "Remove and destroy infected plants.",
            "Plant resistant tomato varieties.",
            "Use reflective mulches to deter whiteflies."
        ]
    }
}


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def preprocess_image(filepath):
    """Preprocess the image for model prediction."""
    # Load the image using TensorFlow's load_img function
    image = tf.keras.preprocessing.image.load_img(filepath, target_size=(128, 128))
    # Convert the image to an array
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    # Normalize the image array
    input_arr = np.array([input_arr])  # Convert single image to a batch
    print("Preprocessed image shape:", input_arr.shape)
    print("Preprocessed image range:", input_arr.min(), input_arr.max())
    return input_arr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about_us')
def about_us():
    return render_template('Aboutus.html')

@app.route('/project_details')
def project_details():
    return render_template('timeline.html')

@app.route('/motivations')
def motivations():
    return render_template('motivate.html')

@app.route('/contact_us')
def contact_us():
    return render_template('project.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        if 'leaf_image' not in request.files:
            return redirect(request.url)
        file = request.files['leaf_image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Preprocess the image
                input_arr = preprocess_image(filepath)

                # Debugging: Print the shape of the input array
                print("Input array shape:", input_arr.shape)
                # Ensure the input array values are within the expected range
                print("Input array values range:", input_arr.min(), input_arr.max())

                # Make predictions
                predictions = model.predict(input_arr)
                print("Predictions:", predictions)
                result_index = np.argmax(predictions)
                prediction = class_names[result_index]

                # Extract plant and disease names
                plant_name, disease_name = prediction.split('___')

                # Get harmfulness score and recommendations based on the predicted disease
                disease_details = disease_info.get(prediction, {
                    'harmfulness': 'Unknown',
                    'recommendations': ["No specific recommendations available."]
                })
                harmfulness = disease_details['harmfulness']
                recommendations = disease_details['recommendations']

                # Debugging: Print the prediction results
                print("Predicted class index:", result_index)
                print("Plant name:", plant_name)
                print("Disease name:", disease_name)
                print("Harmfulness:", harmfulness)
                print("Recommendations:", recommendations)

                return render_template('test.html', filename=filename, plant_name=plant_name, disease_name=disease_name, harmfulness=harmfulness, recommendations=recommendations)
            except Exception as e:
                print(f"Error processing image: {e}")
                return redirect(url_for('test'))

    return render_template('test.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
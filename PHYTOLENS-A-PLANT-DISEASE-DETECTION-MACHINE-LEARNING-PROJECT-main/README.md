# AI-Food

# **PhytoLens: AI-Powered Plant Disease Prediction**
## Overview

PhytoLens is an advanced machine learning-powered web application designed to identify plant diseases from leaf images. Utilizing state-of-the-art deep learning models, PhytoLens enables farmers, researchers, and agricultural professionals to detect and diagnose plant diseases with high accuracy, aiding in effective disease management.


![Image](https://github.com/user-attachments/assets/cc007f56-aea2-4afd-aa16-9698f1a9a2c2)

![Screenshot 2025-01-31 at 11-19-52 Plant Disease Prediction](https://github.com/user-attachments/assets/a6c3745b-bba5-43a1-b5ce-a63afd334659)


# Features

 1. AI-Powered Disease Detection: Leverages deep learning to identify plant diseases from images.

 2. User-Friendly Interface: Simple and interactive web-based platform.

 3. Fast and Accurate Predictions: Provides real-time results with high precision.

 4. Informative Insights: Displays disease information and suggested remedies.

 5. Secure and Scalable: Built using modern web technologies for seamless performance.

# Technologies Used

Frontend: HTML, CSS, JavaScript

Backend: Flask (Python)

Machine Learning Model: TensorFlow/Keras

Dataset: Trained on labeled plant disease datasets


# Lets have a tour of the website 


![Screenshot 2025-01-31 at 11-20-44 Phytolens - Test Leaf](https://github.com/user-attachments/assets/2612420b-eb56-471e-bce7-526c6f086a68)

--- **The main front page**


![Screenshot_31-1-2025_111617_127 0 0 1](https://github.com/user-attachments/assets/a3eecd99-1004-4e26-bb2f-1d6ee69e8f60)

![Screenshot 2025-01-31 at 11-19-52 Plant Disease Prediction](https://github.com/user-attachments/assets/a6c3745b-bba5-43a1-b5ce-a63afd334659)

on clicking Let's detect it moves to the testing page.

![Screenshot 2025-01-31 at 11-21-23 Phytolens - Test Leaf](https://github.com/user-attachments/assets/466babdb-9ff4-4c8f-b289-72eaf288f291)

a prediction report is created based on the particular disease

![Screenshot 2025-01-31 at 11-21-35 Phytolens - Test Leaf](https://github.com/user-attachments/assets/a7b64ae3-419d-43a5-91ee-73325c89906b)

--- **About Us section**


![Screenshot 2025-01-31 at 11-30-38 About Us PhytoLens](https://github.com/user-attachments/assets/5cb2cabc-b4d3-48a2-a2e5-4c849b08e8a8)

![Screenshot 2025-01-31 at 11-30-21 About Us PhytoLens](https://github.com/user-attachments/assets/948917af-771c-4538-84b4-2b8a9976151f)

--- **Project Details Section**


![Screenshot 2025-01-31 at 11-31-04 Project Details - PhytoLens](https://github.com/user-attachments/assets/e75110e6-6082-4e7b-9085-32ae8fd018e9)

![Screenshot 2025-01-31 at 12-17-49 Project Details - PhytoLens](https://github.com/user-attachments/assets/86081e24-c4e8-4372-8ad0-380d1c692ec7)

--- **Motivation Section**


![Screenshot 2025-01-31 at 11-32-14 About Us PhytoLens](https://github.com/user-attachments/assets/883bf368-c382-43a2-a7d6-515129718461)

![Screenshot 2025-01-31 at 11-32-23 About Us PhytoLens](https://github.com/user-attachments/assets/2c356091-fcb6-4445-aaaf-a54be91aaa11)

![Screenshot 2025-01-31 at 11-32-42 About Us PhytoLens](https://github.com/user-attachments/assets/4a62bc70-8602-483b-9569-5f8aa5221db1)


--- **Contact Us**


![Screenshot_31-1-2025_11296_127 0 0 1](https://github.com/user-attachments/assets/a99b8268-5393-45c9-bf38-533e5af784c1)

![Screenshot 2025-01-31 at 11-33-06 Contact Us PhytoLens](https://github.com/user-attachments/assets/2c1b7f31-098f-4b54-85fa-04c29779fbd3)


--- **Testing page**


![Screenshot 2025-01-31 at 11-20-44 Phytolens - Test Leaf](https://github.com/user-attachments/assets/59e97f7b-e34e-451a-924c-0ab2a71a73a2)

![Screenshot 2025-01-31 at 11-21-23 Phytolens - Test Leaf](https://github.com/user-attachments/assets/7d5c6340-f5cc-4f27-b22f-a448a3933ea1)

![Screenshot 2025-01-31 at 11-20-30 Phytolens - Test Leaf](https://github.com/user-attachments/assets/223c8741-a2ed-4e59-90a5-5d21b4edddf6)

# Project Directory Structure

D:\FINAL_PLANT_DIESEASE\Final

│── static

│   ├── css

│   ├── js

│── templates

│   ├── index.html

│   ├── project.html

│   ├── motivate.html

│   ├── test.html

│   ├── timeline.html

│   ├── Aboutus.html


│── venv (virtual environment)


│── trained_plant_disease_model.keras


│── app.py (Main application file)

# Installation & Setup

## Prerequisites

Ensure you have the following installed:

Python (>=3.8)

Flask

TensorFlow/Keras

OpenCV (for image processing)

Git (for version control)


# Steps

### Clone the repository:

git clone https://github.com/your-github-repo/Phytolens.git

### Navigate to the project folder:

cd Phytolens

### Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### Install dependencies:

pip install -r requirements.txt

### Run the application:

python app.py

Access the web app at:   https://phytolens-a-plant-disease-detection-mlp.onrender.com/

# Responsiveness


![Screenshot_31-1-2025_113715_127 0 0 1](https://github.com/user-attachments/assets/b52f9e12-cf74-4e9d-a693-793c98bc0174)

![Screenshot_31-1-2025_113619_127 0 0 1](https://github.com/user-attachments/assets/3cc17805-86d3-46f6-88f8-f281e66e3ccd)

# Usage

Upload a clear image of the affected plant leaf.

Click Predict to analyze the image.

View the disease diagnosis and recommended treatments.

# Future Enhancements

Integration with a mobile application.

Support for multiple plant species.

AI-based treatment recommendations.

Cloud deployment for scalability.

# Contributing

Contributions are welcome! Feel free to fork the repository, create issues, and submit pull requests.

# License

This project is licensed under the MIT License.

# For more details, contact Rani Soni or visit the project repository.
Email id : ranisoni6298@gmail.com


>>>>>>> 1de7b58 (Initial commit)

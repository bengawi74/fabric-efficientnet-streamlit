# Fabric Classification with EfficientNet-B0 and Streamlit

This project was completed as part of the Applied AI Solutions Development program.  
The goal was to build a small image classification system that can identify the type of fabric in an uploaded image.  
The model was trained on a dataset of labeled fabric images and then connected to a simple Streamlit application for testing.

This project helped me practice model training, evaluation, and creating a basic interface that shows predictions clearly.

---

## Project Description

The system uses a pretrained EfficientNet-B0 model as the base, and the final layers were fine-tuned using a fabric image dataset.  
The Streamlit application allows the user to upload an image and receive a prediction along with the model's confidence values.

The project demonstrates the full process from preparing the dataset, training the model, saving the weights, and running a small demo.

---

## Repository Contents

fabric-efficientnet-streamlit  
- models: saved model weights  
- notebooks: training and evaluation notebook  
- app.py: Streamlit application for testing images  
- requirements.txt: list of required Python packages  
- README.md: project documentation

---

## How to Run the Application

1. Clone this repository:

git clone https://github.com/bengawi74/fabric-efficientnet-streamlit.git
cd fabric-efficientnet-streamlit

2. Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate # macOS and Linux
.venv\Scripts\activate # Windows

3. Install the required packages:

pip install -r requirements.txt

4. Run the Streamlit application:

streamlit run app.py

After running the application, a link will appear in the terminal.  
Open it in the browser and upload an image to test the model.

---

## What I Learned

I learned how to fine-tune a pretrained computer vision model, how to organize a project with clear files and folders, how to save and load model weights, and how to create a simple interface for testing the model.  
This project also helped me understand the steps involved in turning a trained model into something a user can actually try.

---

## Purpose of This Project

Classifying fabric types can be useful in areas such as retail, laundry systems, material inspection, and clothing analysis.  
This project allowed me to practice fundamental deep learning concepts using a realistic problem and create a clear and simple demonstration.

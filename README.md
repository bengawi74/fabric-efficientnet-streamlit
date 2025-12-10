# Fabric Classification with EfficientNet-B0

This project is part of my Applied AI Solutions Development program.  
It uses a fine-tuned EfficientNet-B0 model to classify fabric types such as cotton, denim, leather, linen, polyester, and silk.  
I built this project to understand image classification, training pipelines, and how to connect a model to a simple interface.

---

## Project Description

The model was trained on a cleaned dataset of fabric images.  
I used EfficientNet-B0 because it provides good accuracy with a small model size.  
After training, I created a Streamlit app that allows a user to upload an image and receive a prediction with a confidence score.

This project helped me understand:
- how to prepare image datasets  
- how to fine-tune pre-trained models  
- how to evaluate accuracy and loss  
- how to build a small interface for testing a computer vision model

---

## How to Run the Project

1. Install the required packages:

pip install -r requirements.txt

2. Run the Streamlit app:

streamlit run streamlit_app.py

A browser window will open where you can upload an image and see the prediction.

---

## Files in This Repository

- `model/` – trained model weights  
- `notebooks/` – training and evaluation work  
- `streamlit_app.py` – user interface  
- `data/` – dataset structure (directories only)  
- `README.md` – project explanation

---

## What I Learned

This project strengthened my understanding of computer vision workflows.  
I learned how to train a model, save weights, organize files, and present results clearly through a simple interface.  
It also improved my skills in PyTorch, Streamlit, and model evaluation.

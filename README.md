# 🧵 Fabric Classification with EfficientNet & ConvNeXt

A deep-learning image classification system that predicts fabric types — cotton, denim, leather, linen, polyester, and silk.  
This project includes training notebooks, model evaluation, and a Streamlit web app for real-time predictions.

---

## Project Overview

This project builds an image classification system that can recognize different fabric materials. The goal is to help users quickly identify whether an image contains cotton, denim, leather, linen, polyester, or silk. I trained two deep learning models, EfficientNet and ConvNeXt, and compared their performance. The repo also includes a Streamlit app that allows users to upload an image and see the predicted fabric type.


---

## Project Goals

The main goals of this work were:

• Build a clean and organized dataset for fabric images  
• Train two models, EfficientNet and ConvNeXt, and study how they perform  
• Compare accuracy, loss, and confusion matrices  
• Create a simple Streamlit app so anyone can test the model by uploading an image  
• Practice a full workflow from data preparation to model deployment


---

## Repository Structure

The project is organized into a few main parts to make it easy to follow:

• notebooks — training notebooks for EfficientNet and ConvNeXt  
• models — saved model weights  
• app — Streamlit files for running the demo  
• data — folder for images (not included in the repo for size reasons)  
• README — project explanation and instructions


---

## Training and Evaluation

I trained two models for this project. The first one was EfficientNet-B0 and the second was ConvNeXt. Both models were trained on the same set of fabric images so I could compare how they perform under the same conditions.

The models were trained for several epochs, and I tracked accuracy, loss, and confusion matrices to understand how well each model learned the fabric classes. I also reviewed misclassified images to see where the models struggled. The final results showed that both models were able to learn the fabric types, with differences in accuracy depending on the class.


---

## How to Run the Streamlit App

The project includes a small Streamlit app that lets you upload an image and see the predicted fabric type. To run it on your machine, follow these steps:

1. Open the project folder in your terminal.
2. Make sure you have the required libraries installed (Streamlit, PyTorch, Pillow, etc.).
3. Run the app with the command below:

streamlit run app/app.py

4. Your browser will open automatically and you can upload an image to test the model.


---

## Results

Both models were able to classify the fabric images with solid performance. Each model had strengths on different classes, and the confusion matrices helped show exactly where mistakes happened. The results confirmed that the models learned the patterns in the images and were able to predict the correct fabric type most of the time. The Streamlit demo also works smoothly and gives quick predictions on new images.

## What I Learned

From this project, I practiced the full process of building an image classification system. This included preparing the dataset, training two deep learning models, reviewing evaluation metrics, and building a small interactive app. It gave me a better understanding of how different models behave and how to turn a trained model into something that others can use.


# Handwritten Marathi Character Image Recognition

## Overview

This repository contains the code and resources for the project on **Handwritten Marathi Character Image Recognition**, aimed at recognizing handwritten Marathi letters using machine learning and deep learning techniques. The project was developed as part of the academic curriculum at Shivaji University, Kolhapur.

## Project Description

The project focuses on building an AI-based system for recognizing handwritten Marathi characters (अ, आ, इ, ई, उ, ऊ) using various image processing and machine learning techniques. We experimented with traditional models like K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Decision Trees, and then implemented deep learning models like Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) for improved accuracy.

### Key Techniques Used:
- **Image Preprocessing**: Resizing, grayscale conversion, binary conversion, and normalization.
- **Feature Extraction**: Applied techniques tailored to the unique characteristics of Marathi script.
- **Classification Models**: KNN, SVM, Decision Tree, Random Forest, ANN, CNN (LeNet, AlexNet).

## Dataset

The dataset consists of 900 images of handwritten Marathi characters. Each character was collected from handwritten sheets scanned into digital format. The dataset is split into 720 images for training and 180 images for testing.

## Models and Accuracy

We tested several machine learning models, achieving varying levels of accuracy:
- **KNN**: 66.37%
- **SVM**: 83.43%
- **Decision Tree**: 66.72%
- **Random Forest**: 80.02%
- **ANN**: 52%
- **CNN (LeNet)**: 93.78%
- **CNN (AlexNet)**: 95.85%

CNN models, particularly AlexNet, provided the highest accuracy.

## Code

The repository includes implementations for:
1. **Data Preprocessing**: Resizing and formatting the images for model training.
2. **Model Training**: Code for training the models, including KNN, SVM, Decision Trees, Random Forest, ANN, and CNN (LeNet, AlexNet).
3. **Testing and Evaluation**: Evaluating model performance on the test dataset.

## Results

The project successfully demonstrated that Convolutional Neural Networks (CNNs) are the most effective model for handwritten Marathi character recognition, achieving an accuracy of over 95% with AlexNet.

## Conclusion

This project highlights the potential of AI and deep learning in enhancing recognition systems for regional scripts like Marathi. Future work could involve expanding the dataset and exploring more advanced neural network architectures.

## Contributors

- **Karaval Avadhut Mohan**

Under the guidance of:
- **Prof. Dr. D.T. Shirke**
- **Dr. S.D. Pawar**
- **Dr. Sayaji Hande**

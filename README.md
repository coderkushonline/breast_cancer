# Breast Cancer Prediction Using Neural Networks

## Overview
This project involves using a neural network model to predict whether a tumor is malignant or benign based on features extracted from a breast cancer dataset. The dataset is balanced by oversampling the minority class to ensure the model is trained on an even distribution of data.

The project utilizes the **TensorFlow** library to build a neural network for binary classification. The code includes data preprocessing, model training, and prediction on a single input data point.

## Project Structure
The project includes the following steps:
1. **Data Collection and Preprocessing**: Loading the breast cancer dataset, checking for null values, and balancing the dataset.
2. **Feature Scaling**: Standardizing the features to ensure the neural network performs optimally.
3. **Building the Neural Network**: Defining the neural network architecture with layers, compiling the model, and training it.
4. **Prediction**: Making predictions based on the trained model.

## Requirements
- Python 3.x
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Installation
To run the project, make sure you have Python 3.x installed, and install the required libraries by running:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib

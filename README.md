# Automated Pneumonia Detection from Chest X-Rays

## Project Title

Automated Pneumonia Detection from Chest X-Rays using Transfer Learning with ResNet-50.

## Author

MOULYA PERUMAL .U

## Project Description

This project uses Deep Learning and Transfer Learning to classify chest X-ray images into two categories:

1. NORMAL
2. PNEUMONIA

A pretrained ResNet-50 architecture is used as the feature extraction model. The final classification layer is modified for binary classification.

The application is developed using Streamlit.

## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet-50
- Transfer Learning
- Streamlit
- Pillow
- Pandas
- Power BI

## Project Workflow

Chest X-Ray
      ↓
Image Upload
      ↓
Image Resizing
      ↓
Pixel Normalization
      ↓
ResNet-50
      ↓
Feature Extraction
      ↓
Classification Layer
      ↓
Normal / Pneumonia
      ↓
Confidence Score
      ↓
CSV
      ↓
Power BI Dashboard

## Model

The project uses pretrained ResNet-50.

Input image size:

224 × 224 pixels

The final ResNet classification layer is replaced with a custom classifier for two classes.

## Classes

NORMAL

PNEUMONIA

## Frontend

The Streamlit application contains:

- Animated background
- Glassmorphism cards
- X-ray image upload
- Image preview
- Prediction button
- Prediction result
- Confidence score
- Class probability bars
- Prediction history
- CSV download

## Power BI

Prediction results are stored in:

predictions.csv

The CSV file can be imported into Microsoft Power BI to create analytics dashboards.

Possible Power BI visuals:

- Total Predictions
- Normal Count
- Pneumonia Count
- Prediction Distribution
- Confidence Distribution
- Prediction Timeline


## Image of website 
<img width="1906" height="857" alt="Front page of website " src="https://github.com/user-attachments/assets/024259a7-41db-469e-9577-82a67de5dad2" />
<img width="1913" height="780" alt="Prediction analysis" src="https://github.com/user-attachments/assets/a4049617-9eb2-4a08-8ab2-b63934636103" />
<img width="1916" height="858" alt="Input and Output " src="https://github.com/user-attachments/assets/eb97eebb-d82c-4275-b6d8-78481eae9352" />


## Disclaimer

This project is intended for educational and demonstration purposes only.

The model output must not be treated as a medical diagnosis. Medical decisions should be made by qualified healthcare professionals.

# Fashion MNIST Image Classification Project

This project provides a complete pipeline for training, evaluating, and deploying a machine learning model for fashion item classification using the Fashion MNIST dataset. The project includes data preprocessing, model training, evaluation, Grad-CAM visualization, and an API for predictions.

## Requirements

- Python version <= 3.10.11
- Install dependencies using `requirements.txt`:
  ```bash
  pip install -r requirements.txt
## Training
- Run the training script to train the MobileNetV2 model on the Fashion MNIST dataset:

  ```bash
  python fashion_mnist_training.py

After running the training script, it will generate x_test.npy and y_test.npy files for test data.

## Evaluation
- Run the evaluation script on the generated test data to produce a classification report and confusion matrix:

  ```bash
  python fashion_mnist_evaluation.py
## Grad-CAM Visualization
- Run the Grad-CAM script to generate a heatmap on the given input test image:

  ```bash
  python grad_cam_heatmap.py -i <image_path>
Replace <image_path> with the path to the input image.

## Running the Prediction API

## Azure deployed web app
```bash
https://fashionimageprediction-hza3fzava4hwcbak.canadacentral-01.azurewebsites.net/
```
## Backend
- To run the FastAPI backend for the prediction API, use the following command:

  ```bash
  uvicorn app:app --reload --host 0.0.0.0 --port 8001
## Frontend
- To run the Streamlit frontend for the prediction API, use the following command:

  ```bash
  streamlit run frontend.py
## Testing the Backend API using Curl Test Scripts
- You can test the backend FastAPI endpoints using the provided PowerShell scripts.

Run the FastAPI Server
  ```bash
  uvicorn app:app --reload --host 0.0.0.0 --port 8001
```
- Test Scripts

In a new terminal, navigate to the curl_test_scripts directory and run the following three scripts to test the backend endpoints:

- Register Endpoint:

  ```powershell
  .\register.ps1
- Sign-In Endpoint:

  ```powershell
  .\signin.ps1
- Predict Endpoint:

  ```powershell      
  .\predict.ps1

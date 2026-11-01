# Plasmo - Clinical Malaria Parasite Diagnostic Workstation

This is a deep convolutional neural network workstation that processes microscopic blood smear cell images to classify malaria pathology. It provides diagnostic outputs alongside **Grad-CAM visual localizations** from the last convolutional layer (`Conv2D_2` with 128 filters) to display target pathology focus.

## Tech Stack
* **Backend:** FastAPI
* **Deep Learning Framework:** TensorFlow / Keras 3.x
* **Computer Vision:** OpenCV, Pillow
* **Client Interface Layer:** CORS enabled for direct cross-origin browser communications

## Decision Rule Metrics
* Sigmoid Activation Layer Threshold: `>= 0.50` represents **Uninfected**, `< 0.50` represents **Parasitized**.

## How to Run Locally

1. Install requirements:
   ```bash
   pip install fastapi uvicorn tensorflow opencv-python pillow numpy
   ```
2. Place your trained model file (`malaria_model.keras`) in the root directory.
3. Start the local server endpoint:
   ```bash
   uvicorn main:app --reload
   ```

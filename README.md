# Brain Tumor Detection Web Application

This project is an advanced, research-grade web application for Brain Tumor Detection. It utilizes a deep learning model (VGG16 Transfer Learning) to classify MRI scans into four categories: **Glioma, Meningioma, No Tumor, and Pituitary**. 

The application goes beyond simple classification by providing:
- **Grad-CAM Explainability:** Generates a heatmap showing which parts of the MRI scan the model focused on.
- **Bayesian Uncertainty Quantification:** Calculates the model's confidence and reliability.
- **Advanced 6-Panel Visualization:** Displays original scans, heatmaps, probability distributions, and uncertainty metrics.
- **Research Analytics & Data Export:** Logs predictions and allows data export for research purposes.

---

## Prerequisites

Before running the project, ensure you have the following installed on your system:
- **Python** (version 3.8 to 3.11 recommended)
- **Git**
- **Git LFS (Large File Storage):** Required to download the pre-trained model file (`brain_tumor_model.h5`).

---

## Step-by-Step Installation and Setup

Follow these exact steps to run the application on your local machine.

### 1. Clone the Repository
Open your terminal (Command Prompt, PowerShell, or Git Bash) and clone this repository to your local machine:
```bash
git clone https://github.com/ProgrammerGuy3009/Brain_Tumour_Detection_V1.git
cd Brain_Tumour_Detection_V1
```

### 2. Pull Git LFS Files
Because the pre-trained model (`brain_tumor_model.h5`) is larger than 100MB, it is stored using Git LFS. You need to pull the actual model file:
```bash
git lfs install
git lfs pull
```
*(Verify that the file `brain_tumor_model.h5` is around 111MB. If it is only a few KBs, the LFS pull was not successful.)*

### 3. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python projects.
```bash
# For Windows
python -m venv venv

# For macOS/Linux
python3 -m venv venv
```

### 4. Activate the Virtual Environment
Activate the environment you just created.
```bash
# For Windows (Command Prompt)
venv\Scripts\activate.bat

# For Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# For macOS/Linux
source venv/bin/activate
```
*(Once activated, you should see `(venv)` at the beginning of your terminal prompt.)*

### 5. Install Dependencies
With the virtual environment activated, install all the required Python libraries listed in the `requirements.txt` file. We will also install `matplotlib` which is used for generating the visualizations.
```bash
pip install -r requirements.txt
pip install matplotlib
```

### 6. Run the Application
Finally, start the Flask web server by running the main Python script:
```bash
python app-debug.py
```
You should see output in the terminal indicating the model has loaded and the server has started.

### 7. Access the Web App
Open your favorite web browser and navigate to:
```
http://localhost:5000
```

---

## How to Use

1. **Upload an MRI Scan:** On the home page, click to upload an MRI scan image (JPG, PNG).
2. **Predict:** Click the "Predict" button.
3. **View Results:** The application will process the image and return a detailed, 6-panel visualization that includes:
   - The predicted tumor class.
   - The Grad-CAM heatmap showing the area of interest.
   - The probability breakdown of all classes.
   - Uncertainty metrics (Confidence, Epistemic/Aleatoric Uncertainty, Model Reliability).
   - A clinical recommendation based on the model's confidence.

## Research Endpoints

For research purposes, the application includes additional endpoints:
- **`/analytics`**: Returns JSON data with aggregated statistics on all predictions made.
- **`/export_research_data`**: Downloads a CSV file containing the complete log of all predictions and uncertainty metrics.
- **`/model_info`**: Provides detailed metadata about the model architecture and features.

## Troubleshooting

- **Error: "Model not found" or "h5py error"**: Make sure you ran `git lfs pull`. If the `.h5` file is tiny (under 1MB), you downloaded the Git LFS pointer file, not the actual model.
- **Error: "ModuleNotFoundError"**: Ensure your virtual environment is activated and you ran `pip install -r requirements.txt`.

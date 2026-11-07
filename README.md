# Breast Cancer Wisconsin Classifier

An educational machine-learning assignment comparing Logistic Regression,
Decision Tree, KNN, Gaussian Naive Bayes, and SVM on the UCI Breast Cancer
Wisconsin (Diagnostic) dataset.

## Run the project

1. Create a Python virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Activate it:

   macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   Windows Command Prompt:

   ```cmd
   .venv\Scripts\activate.bat
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app from this project directory:

   ```bash
   streamlit run app.py
   ```

## App usage

The app displays the model comparison metrics and lets you choose a trained
model to classify rows uploaded as a CSV. Download the provided CSV template
to get the expected feature column names. Do not include an ID or diagnosis
column; provide one row of numeric feature values per sample.

The model was built for an educational assignment only. It is not a medical
device and must not be used for diagnosis or treatment decisions.
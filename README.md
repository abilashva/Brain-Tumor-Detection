# Brain Tumor Detection System

A CNN-based image classifier that detects the presence of a tumor in brain
MRI scans, deployed through an interactive Streamlit interface for
real-time image upload and prediction.

**Tech stack:** Python, TensorFlow/Keras, Streamlit, NumPy, Pillow, scikit-learn

## Results

- **Validation accuracy: 79.59%**
- The model is stronger at detecting tumors (90% recall on "yes") than
  confirming their absence (53% recall on "no tumor"), likely due to class
  imbalance in the training data (154 tumor images vs. 96 non-tumor images).
- A natural next step to improve this would be class weighting during
  training, or gathering a more balanced dataset.

## Dataset

Source: [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

This is a **binary classification** problem — two classes only:

brain_tumor_dataset/
yes/   -> MRI images with a tumor
no/    -> MRI images without a tumor

Download the dataset from Kaggle and place it in this folder before training.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

## Train the model
python train.py

This will:
- Train a CNN (3 convolutional blocks + dense classifier head) from scratch
- Print per-epoch training/validation accuracy
- Print a final classification report and confusion matrix
- Save the trained model as brain_tumor_model.h5

### No local GPU? Use Google Colab (free)
Open Google Colab
Upload train.py and the brain_tumor_dataset/ folder (or mount Google Drive)
Run in a cell:
!pip install -r requirements.txt
!python train.py
Download the resulting brain_tumor_model.h5 back to your machine

## Run the app
streamlit run app.py
Upload an MRI image in the browser and get a real-time Tumor / No Tumor
prediction with a confidence score.
## Notes
This project is for educational purposes only and is not a medical
diagnostic tool.
If deploying publicly (e.g. Streamlit Community Cloud), add the live
URL here once available.

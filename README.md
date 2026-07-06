# VK Gender Prediction — Setup Guide

## Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Add Your Dataset
Download the dataset from AllCups platform and place files in the `data/` folder:
```
data/
├── train.csv    ← training data with gender labels
└── test.csv     ← test data for final predictions (optional)
```

> **⚠️ Important:** Update `TARGET_COL` in the notebook/script if the target column is named differently (e.g., `gender`, `sex`, `label`).

### Step 3: Run the Notebook
```bash
jupyter notebook VK_Gender_Prediction.ipynb
```

Or run as a Python script:
```bash
python gender_prediction.py
```

### Step 4: Check Outputs
All results are saved to:
- `models/` — trained model (.joblib files)
- `output/` — plots and CSV metrics

---

## Packaging for Submission

After running the notebook, create the submission archive:

```bash
# Windows PowerShell
Compress-Archive -Path VK_Gender_Prediction.ipynb, gender_prediction.py, REPORT.md, models\, output\ -DestinationPath solution.zip
```

Then upload `solution.zip` to **Cloud Mail** (cloud.mail.ru) and share the link.

---

## Cloud Mail (облако Mail.ru) — Instructions

1. Go to **cloud.mail.ru** and sign in (or register with a Mail.ru / VK account)
2. Click **"Загрузить"** (Upload) and select `solution.zip`
3. Wait for upload to complete
4. Hover over the file → click **"Получить ссылку"** (Get link)
5. Enable public access and copy the link
6. Submit the link to the feedback form on the learning platform

---

## Project Structure
```
vk-gender-prediction/
├── VK_Gender_Prediction.ipynb   ← Main Jupyter Notebook
├── gender_prediction.py          ← Python script (same logic)
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
├── REPORT.md                     ← Project report
├── data/                         ← Put your CSV files here
├── models/                       ← Auto-created; stores .joblib model
└── output/                       ← Auto-created; stores plots & CSVs
```
